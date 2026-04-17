use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

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

#[derive(Debug, Deserialize, Serialize)]
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
struct ProjectSparseInput {
    shape: Vec<usize>,
    slots: Vec<usize>,
    max_terms: Option<usize>,
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

#[derive(Debug, Deserialize)]
struct BenchmarkManifest {
    canonicalization_cases: Vec<BenchmarkCase<CanonicalizeInput>>,
    projection_cases: Vec<BenchmarkCase<ProjectSparseInput>>,
    decomposition_cases: Vec<BenchmarkCase<LrDecomposeInput>>,
}

#[derive(Debug, Deserialize)]
struct BenchmarkCase<T> {
    name: String,
    repetitions: usize,
    payload: T,
}

fn build_multiterm_identity(name: &str) -> Result<ax_ir::TensorMultitermIdentity> {
    match name {
        "first_bianchi" => Ok(ax_ir::TensorMultitermIdentity::FirstBianchi {
            cyclic_slots: [1, 2, 3],
        }),
        other => anyhow::bail!("unsupported oracle multiterm identity: {other}"),
    }
}

fn render_sparse_projection(input: ProjectSparseInput) -> Result<String> {
    let diagram = ax_young::YoungDiagram::try_new(input.shape)?;
    let tableau = ax_young::YoungTableau::standard(&diagram)?;
    let projector = ax_young::build_group_backed_projector(
        &tableau,
        ax_young::ProjectorNormalization::Unnormalized,
    )?;
    let plan = ax_young::build_sparse_projector_plan(&projector)?;
    let (terms, _) =
        ax_young::apply_sparse_plan_to_slots(&plan, &input.slots, input.max_terms.unwrap_or(64))?;
    let rendered = terms
        .into_iter()
        .map(|(slots, coeff)| format!("{}/{}:{slots:?}", coeff.numer(), coeff.denom()))
        .collect::<Vec<_>>()
        .join("|");
    Ok(rendered)
}

fn run_young_case(case: YoungOracleCase) -> ax_trace::OracleCaseTrace {
    let (expected, actual, passed) = match case.kind.as_str() {
        "canonicalize" => {
            let expected_value: Result<CanonicalizeExpected> =
                serde_json::from_value(case.expected.clone()).map_err(anyhow::Error::from);
            let input_value: Result<CanonicalizeInput> =
                serde_json::from_value(case.input.clone()).map_err(anyhow::Error::from);
            match (input_value, expected_value) {
                (Ok(input), Ok(expected_case)) => {
                    let expected_string = serde_json::json!({
                        "canonical": expected_case.canonical,
                        "sign": expected_case.sign,
                    })
                    .to_string();
                    let actual_result = ax_tensor::oracle_canonicalize_string(
                        &input.shape,
                        &input.slot_map,
                        &input.slots,
                    )
                    .map(|(canonical, sign)| {
                        serde_json::json!({
                            "canonical": canonical,
                            "sign": sign.to_string(),
                        })
                        .to_string()
                    });
                    match actual_result {
                        Ok(actual_string) => {
                            let passed = actual_string == expected_string;
                            (expected_string, actual_string, passed)
                        }
                        Err(err) => (expected_string, format!("error: {err:#}"), false),
                    }
                }
                (Err(err), _) | (_, Err(err)) => {
                    ("parse_error".to_string(), format!("error: {err:#}"), false)
                }
            }
        }
        "project_sparse" => {
            let expected_string = case.expected.to_string();
            let input_value: Result<ProjectSparseInput> =
                serde_json::from_value(case.input.clone()).map_err(anyhow::Error::from);
            match input_value.and_then(render_sparse_projection) {
                Ok(actual_string) => {
                    let passed = actual_string == expected_string;
                    (expected_string, actual_string, passed)
                }
                Err(err) => (expected_string, format!("error: {err:#}"), false),
            }
        }
        "lr_decompose" => {
            let expected_value: Result<LrDecomposeExpected> =
                serde_json::from_value(case.expected.clone()).map_err(anyhow::Error::from);
            let input_value: Result<LrDecomposeInput> =
                serde_json::from_value(case.input.clone()).map_err(anyhow::Error::from);
            match (input_value, expected_value) {
                (Ok(input), Ok(expected_case)) => {
                    let expected_string = serde_json::json!({
                        "shapes": expected_case.shapes,
                        "multiplicities": expected_case.multiplicities,
                    })
                    .to_string();
                    match ax_tensor::oracle_lr_decompose(&input.factors) {
                        Ok((shapes, multiplicities)) => {
                            let actual_string = serde_json::json!({
                                "shapes": shapes,
                                "multiplicities": multiplicities,
                            })
                            .to_string();
                            let passed = actual_string == expected_string;
                            (expected_string, actual_string, passed)
                        }
                        Err(err) => (expected_string, format!("error: {err:#}"), false),
                    }
                }
                (Err(err), _) | (_, Err(err)) => {
                    ("parse_error".to_string(), format!("error: {err:#}"), false)
                }
            }
        }
        "character" => {
            let expected_value: Result<CharacterExpected> =
                serde_json::from_value(case.expected.clone()).map_err(anyhow::Error::from);
            let input_value: Result<CharacterInput> =
                serde_json::from_value(case.input.clone()).map_err(anyhow::Error::from);
            match (input_value, expected_value) {
                (Ok(input), Ok(expected_case)) => {
                    let expected_string = expected_case.character;
                    let actual_result =
                        (|| -> Result<String> {
                            let diagram = ax_young::YoungDiagram::try_new(input.shape)?;
                            Ok(ax_young::symmetric_group_character(&diagram, &input.cycle)?
                                .to_string())
                        })();
                    match actual_result {
                        Ok(actual_string) => {
                            let passed = actual_string == expected_string;
                            (expected_string, actual_string, passed)
                        }
                        Err(err) => (expected_string, format!("error: {err:#}"), false),
                    }
                }
                (Err(err), _) | (_, Err(err)) => {
                    ("parse_error".to_string(), format!("error: {err:#}"), false)
                }
            }
        }
        "multiterm_reduce" => {
            let expected_value: Result<MultitermReduceExpected> =
                serde_json::from_value(case.expected.clone()).map_err(anyhow::Error::from);
            let input_value: Result<MultitermReduceInput> =
                serde_json::from_value(case.input.clone()).map_err(anyhow::Error::from);
            match (input_value, expected_value) {
                (Ok(input), Ok(expected_case)) => {
                    let expected_string = expected_case.reduced;
                    let actual_result =
                        build_multiterm_identity(&input.identity).and_then(|identity| {
                            ax_tensor::oracle_multiterm_reduce_string(
                                &input.symbol,
                                &input.slots,
                                &identity,
                            )
                        });
                    match actual_result {
                        Ok(actual_string) => {
                            let passed = actual_string == expected_string;
                            (expected_string, actual_string, passed)
                        }
                        Err(err) => (expected_string, format!("error: {err:#}"), false),
                    }
                }
                (Err(err), _) | (_, Err(err)) => {
                    ("parse_error".to_string(), format!("error: {err:#}"), false)
                }
            }
        }
        _ => (
            "unsupported_kind".to_string(),
            format!("error: unsupported oracle case kind {}", case.kind),
            false,
        ),
    };

    ax_trace::OracleCaseTrace {
        case_name: case.name,
        kind: case.kind,
        expected,
        actual,
        passed,
    }
}

fn run_curvature_case(case: CurvatureOracleCase) -> ax_trace::OracleCaseTrace {
    let expected_string = serde_json::json!(case.expected_terms).to_string();
    let actual_result =
        ax_tensor::oracle_curvature_terms(&case.input_kind, case.dimension).map(|terms| {
            serde_json::json!(terms
                .into_iter()
                .map(|(kind, numer, denom)| serde_json::json!({
                    "kind": kind,
                    "numer": numer,
                    "denom": denom,
                }))
                .collect::<Vec<_>>())
            .to_string()
        });

    let (actual, passed) = match actual_result {
        Ok(actual_string) => {
            let passed = actual_string == expected_string;
            (actual_string, passed)
        }
        Err(err) => (format!("error: {err:#}"), false),
    };

    ax_trace::OracleCaseTrace {
        case_name: case.name,
        kind: case.input_kind,
        expected: expected_string,
        actual,
        passed,
    }
}

pub fn run_oracle_corpus(path: &Path) -> anyhow::Result<Vec<ax_trace::OracleCaseTrace>> {
    let run = || -> anyhow::Result<Vec<ax_trace::OracleCaseTrace>> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let value: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        let items = value
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("oracle corpus must be a top-level array"))?;

        let mut traces = Vec::with_capacity(items.len());
        for item in items {
            if item.get("kind").is_some() {
                let case: YoungOracleCase = serde_json::from_value(item.clone())?;
                traces.push(run_young_case(case));
            } else {
                let case: CurvatureOracleCase = serde_json::from_value(item.clone())?;
                traces.push(run_curvature_case(case));
            }
        }
        Ok(traces)
    };

    run().context("failed to run oracle corpus")
}

pub fn run_benchmark_manifest(path: &Path) -> anyhow::Result<Vec<(String, usize)>> {
    let run = || -> anyhow::Result<Vec<(String, usize)>> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let manifest: BenchmarkManifest = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse {}", path.display()))?;

        let mut results = Vec::new();

        for case in manifest.canonicalization_cases {
            for _ in 0..case.repetitions {
                let _ = ax_tensor::oracle_canonicalize_string(
                    &case.payload.shape,
                    &case.payload.slot_map,
                    &case.payload.slots,
                )?;
            }
            results.push((case.name, case.repetitions));
        }

        for case in manifest.projection_cases {
            for _ in 0..case.repetitions {
                let _ = render_sparse_projection(ProjectSparseInput {
                    shape: case.payload.shape.clone(),
                    slots: case.payload.slots.clone(),
                    max_terms: case.payload.max_terms,
                })?;
            }
            results.push((case.name, case.repetitions));
        }

        for case in manifest.decomposition_cases {
            for _ in 0..case.repetitions {
                let _ = ax_tensor::oracle_lr_decompose(&case.payload.factors)?;
            }
            results.push((case.name, case.repetitions));
        }

        Ok(results)
    };

    run().context("failed to run benchmark manifest")
}
