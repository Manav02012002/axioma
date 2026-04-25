use crate::Env;
use ax_ir::Expr;
use std::collections::HashMap;

pub struct InspectResult {
    pub kind: String,
    pub free_indices: Vec<(String, String)>,
    pub dummy_pairs: Vec<(String, String)>,
    pub properties: Vec<(String, Vec<String>)>,
    pub symbols: Vec<String>,
    pub node_count: usize,
}

fn expr_kind(expr: &Expr) -> &'static str {
    match expr {
        Expr::Indexed(_, _) => "indexed",
        Expr::Matrix(_) => "matrix",
        Expr::List(_) => "list",
        Expr::Add(_) => "sum",
        Expr::Mul(_) => "product",
        Expr::Call(_, _) => "function_call",
        Expr::Int(_) | Expr::Rational(_) | Expr::Float(_) | Expr::Sym(_) => "scalar",
        _ => "other",
    }
}

fn collect_indices(expr: &Expr, out: &mut Vec<ax_ir::Index>) {
    match expr {
        Expr::Indexed(base, indices) => {
            collect_indices(base, out);
            out.extend(indices.iter().cloned());
        }
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            for term in terms {
                collect_indices(term, out);
            }
        }
        Expr::Matrix(rows) => {
            for row in rows {
                for cell in row {
                    collect_indices(cell, out);
                }
            }
        }
        Expr::Pow(base, exp) => {
            collect_indices(base, out);
            collect_indices(exp, out);
        }
        Expr::Neg(inner) | Expr::Group(inner, _) => collect_indices(inner, out),
        Expr::Complex(re, im) => {
            collect_indices(re, out);
            collect_indices(im, out);
        }
        Expr::Call(_, args) => {
            for arg in args {
                collect_indices(arg, out);
            }
        }
        Expr::FnDef(_, _, body) => collect_indices(body, out),
        Expr::Rule(lhs, rhs, _) => {
            collect_indices(lhs, out);
            collect_indices(rhs, out);
        }
        Expr::Piecewise(cases) => {
            for (value, _) in cases {
                collect_indices(value, out);
            }
        }
        Expr::Let(_, value, body) => {
            collect_indices(value, out);
            collect_indices(body, out);
        }
        Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::SetConvention(_, _)
        | Expr::Sym(_)
        | Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_) => {}
    }
}

fn collect_symbols(expr: &Expr, out: &mut Vec<ax_ir::expr::Sym>) {
    match expr {
        Expr::Sym(sym) => out.push(*sym),
        Expr::Indexed(base, _) => collect_symbols(base, out),
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            for term in terms {
                collect_symbols(term, out);
            }
        }
        Expr::Matrix(rows) => {
            for row in rows {
                for cell in row {
                    collect_symbols(cell, out);
                }
            }
        }
        Expr::Pow(base, exp) => {
            collect_symbols(base, out);
            collect_symbols(exp, out);
        }
        Expr::Neg(inner) | Expr::Group(inner, _) => collect_symbols(inner, out),
        Expr::Complex(re, im) => {
            collect_symbols(re, out);
            collect_symbols(im, out);
        }
        Expr::Call(_, args) => {
            for arg in args {
                collect_symbols(arg, out);
            }
        }
        Expr::FnDef(_, _, body) => collect_symbols(body, out),
        Expr::Rule(lhs, rhs, _) => {
            collect_symbols(lhs, out);
            collect_symbols(rhs, out);
        }
        Expr::Piecewise(cases) => {
            for (value, _) in cases {
                collect_symbols(value, out);
            }
        }
        Expr::Let(_, value, body) => {
            collect_symbols(value, out);
            collect_symbols(body, out);
        }
        Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::SetConvention(_, _)
        | Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_) => {}
    }
}

fn node_count(expr: &Expr) -> usize {
    match expr {
        Expr::Int(_)
        | Expr::Rational(_)
        | Expr::Float(_)
        | Expr::Sym(_)
        | Expr::Import(_)
        | Expr::Assume(_, _)
        | Expr::SetConvention(_, _) => 1,
        Expr::Indexed(base, indices) => 1 + node_count(base) + indices.len(),
        Expr::Add(terms) | Expr::Mul(terms) | Expr::List(terms) => {
            1 + terms.iter().map(node_count).sum::<usize>()
        }
        Expr::Matrix(rows) => 1 + rows.iter().flatten().map(node_count).sum::<usize>(),
        Expr::Pow(base, exp) => 1 + node_count(base) + node_count(exp),
        Expr::Neg(inner) => 1 + node_count(inner),
        Expr::Complex(re, im) => 1 + node_count(re) + node_count(im),
        Expr::Call(_, args) => 1 + args.iter().map(node_count).sum::<usize>(),
        Expr::FnDef(_, _, body) => 1 + node_count(body),
        Expr::Rule(lhs, rhs, _) => 1 + node_count(lhs) + node_count(rhs),
        Expr::Piecewise(cases) => {
            1 + cases
                .iter()
                .map(|(value, _)| node_count(value))
                .sum::<usize>()
        }
        Expr::Let(_, value, body) => 1 + node_count(value) + node_count(body),
        Expr::Group(inner, _) => 1 + node_count(inner),
    }
}

fn property_name(prop: &ax_ir::TensorProperty, interner: &ax_ir::Interner) -> String {
    match prop {
        ax_ir::TensorProperty::Symmetric(pos) => format!("Symmetric({pos:?})"),
        ax_ir::TensorProperty::AntiSymmetric(pos) => format!("AntiSymmetric({pos:?})"),
        ax_ir::TensorProperty::RiemannSymmetry => "RiemannSymmetry".to_string(),
        ax_ir::TensorProperty::Traceless => "Traceless".to_string(),
        ax_ir::TensorProperty::Diagonal => "Diagonal".to_string(),
        ax_ir::TensorProperty::Trace => "Trace".to_string(),
        ax_ir::TensorProperty::Metric => "Metric".to_string(),
        ax_ir::TensorProperty::InverseMetric => "InverseMetric".to_string(),
        ax_ir::TensorProperty::KroneckerDelta => "KroneckerDelta".to_string(),
        ax_ir::TensorProperty::EpsilonTensor => "EpsilonTensor".to_string(),
        ax_ir::TensorProperty::Derivative => "Derivative".to_string(),
        ax_ir::TensorProperty::PartialDerivative => "PartialDerivative".to_string(),
        ax_ir::TensorProperty::CovariantDerivative => "CovariantDerivative".to_string(),
        ax_ir::TensorProperty::TableauInherit => "TableauInherit".to_string(),
        ax_ir::TensorProperty::Depends(syms) => format!(
            "Depends({:?})",
            syms.iter()
                .map(|s| interner.resolve(*s).to_string())
                .collect::<Vec<_>>()
        ),
        ax_ir::TensorProperty::Spinor => "Spinor".to_string(),
        ax_ir::TensorProperty::SpinorMeta(metadata) => format!(
            "SpinorMeta(class={:?}, dimension={:?}, chirality={:?}, index_family={:?})",
            metadata.class,
            metadata.dimension,
            metadata.chirality,
            metadata
                .index_family
                .map(|sym| interner.resolve(sym).to_string())
        ),
        ax_ir::TensorProperty::DiracBar => "DiracBar".to_string(),
        ax_ir::TensorProperty::DiracBarMeta(metadata) => format!(
            "DiracBarMeta(gamma_symbol={:?}, spinor_family={:?}, reverse_gamma_order={})",
            metadata
                .gamma_symbol
                .map(|sym| interner.resolve(sym).to_string()),
            metadata
                .spinor_family
                .map(|sym| interner.resolve(sym).to_string()),
            metadata.reverse_gamma_order
        ),
        ax_ir::TensorProperty::GammaMatrixProp => "GammaMatrixProp".to_string(),
        ax_ir::TensorProperty::GammaMatrixMeta(metadata) => format!(
            "GammaMatrixMeta(dimension={:?}, metric_symbol={:?}, index_family={:?}, has_gamma5={})",
            metadata.dimension,
            metadata
                .metric_symbol
                .map(|sym| interner.resolve(sym).to_string()),
            metadata
                .index_family
                .map(|sym| interner.resolve(sym).to_string()),
            metadata.has_gamma5
        ),
        ax_ir::TensorProperty::GammaConventionMeta(metadata) => format!(
            "GammaConventionMeta(signature={:?}, clifford={:?}, gamma5={:?}, epsilon_symbol={:?}, dimension={:?})",
            metadata.signature,
            metadata.clifford,
            metadata.gamma5,
            metadata
                .epsilon_symbol
                .map(|sym| interner.resolve(sym).to_string()),
            metadata.dimension
        ),
        ax_ir::TensorProperty::Commuting => "Commuting".to_string(),
        ax_ir::TensorProperty::AntiCommuting => "AntiCommuting".to_string(),
        ax_ir::TensorProperty::NonCommuting => "NonCommuting".to_string(),
        ax_ir::TensorProperty::CommutingWith(syms) => format!(
            "CommutingWith({:?})",
            syms.iter()
                .map(|s| interner.resolve(*s).to_string())
                .collect::<Vec<_>>()
        ),
        ax_ir::TensorProperty::AntiCommutingWith(syms) => format!(
            "AntiCommutingWith({:?})",
            syms.iter()
                .map(|s| interner.resolve(*s).to_string())
                .collect::<Vec<_>>()
        ),
        ax_ir::TensorProperty::NonCommutingWith(syms) => format!(
            "NonCommutingWith({:?})",
            syms.iter()
                .map(|s| interner.resolve(*s).to_string())
                .collect::<Vec<_>>()
        ),
        ax_ir::TensorProperty::SelfAntiCommuting => "SelfAntiCommuting".to_string(),
        ax_ir::TensorProperty::SelfNonCommuting => "SelfNonCommuting".to_string(),
        ax_ir::TensorProperty::SelfCommuting => "SelfCommuting".to_string(),
        ax_ir::TensorProperty::CommutingAsProduct => "CommutingAsProduct".to_string(),
        ax_ir::TensorProperty::CommutingAsSum => "CommutingAsSum".to_string(),
        ax_ir::TensorProperty::MajoranaSpinor => "MajoranaSpinor".to_string(),
        ax_ir::TensorProperty::WeylSpinor => "WeylSpinor".to_string(),
        ax_ir::TensorProperty::ImplicitIndex => "ImplicitIndex".to_string(),
        ax_ir::TensorProperty::SortOrder(syms) => format!(
            "SortOrder({:?})",
            syms.iter()
                .map(|s| interner.resolve(*s).to_string())
                .collect::<Vec<_>>()
        ),
        ax_ir::TensorProperty::TableauSymmetry(symmetry) => {
            format!("TableauSymmetry(tableaux={:?})", symmetry.tableaux)
        }
        ax_ir::TensorProperty::MixedTableauSymmetry(symmetry) => {
            format!("MixedTableauSymmetry(tableaux={:?})", symmetry.tableaux)
        }
        ax_ir::TensorProperty::GradedParity(values) => format!("GradedParity({values:?})"),
        ax_ir::TensorProperty::TensorIdentities(identities) => {
            format!("TensorIdentities(multiterm={:?})", identities.multiterm)
        }
        ax_ir::TensorProperty::SatisfiesBianchi { slots } => {
            format!("SatisfiesBianchi(slots={slots:?})")
        }
        ax_ir::TensorProperty::DimensionDependentIdentity => {
            "DimensionDependentIdentity".to_string()
        }
        ax_ir::TensorProperty::WeylTensor => "WeylTensor".to_string(),
        ax_ir::TensorProperty::DifferentialFormDegree(n) => {
            format!("DifferentialFormDegree({n})")
        }
        ax_ir::TensorProperty::HilbertSpaceMeta(metadata) => format!(
            "HilbertSpaceMeta(dimension={}, factors=[{}])",
            metadata.dimension,
            metadata
                .factors
                .iter()
                .map(|factor| format!("{}:{}", interner.resolve(factor.symbol), factor.dimension))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        ax_ir::TensorProperty::FockSpaceMeta(metadata) => format!(
            "FockSpaceMeta(symbol={}, modes=[{}], basis_order=[{}])",
            interner.resolve(metadata.symbol),
            metadata
                .modes
                .iter()
                .map(|mode| format!(
                    "{}:{:?}:{:?}",
                    interner.resolve(mode.symbol),
                    mode.statistics,
                    mode.truncation
                ))
                .collect::<Vec<_>>()
                .join(", "),
            metadata
                .basis_order
                .iter()
                .map(|sym| interner.resolve(*sym).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        ax_ir::TensorProperty::QuantumObjectMeta(metadata) => format!(
            "QuantumObjectMeta(kind={:?}, space_symbol={})",
            metadata.kind,
            interner.resolve(metadata.space_symbol)
        ),
        ax_ir::TensorProperty::OperatorSpaceMeta(metadata) => format!(
            "OperatorSpaceMeta(domain_space={}, codomain_space={})",
            interner.resolve(metadata.domain_space),
            interner.resolve(metadata.codomain_space)
        ),
        ax_ir::TensorProperty::ModeMeta(metadata) => format!(
            "ModeMeta(statistics={:?}, subsystem={:?}, mode_index={}, label={:?})",
            metadata.statistics,
            metadata
                .subsystem
                .map(|sym| interner.resolve(sym).to_string()),
            metadata.mode_index,
            metadata.label.map(|sym| interner.resolve(sym).to_string())
        ),
        ax_ir::TensorProperty::BackgroundClass(sym) => {
            format!("BackgroundClass({})", interner.resolve(*sym))
        }
        ax_ir::TensorProperty::PerturbationFamily { family, order } => {
            format!(
                "PerturbationFamily(family={}, order={order})",
                interner.resolve(*family)
            )
        }
        ax_ir::TensorProperty::SectorTag(sym) => {
            format!("SectorTag({})", interner.resolve(*sym))
        }
        ax_ir::TensorProperty::GaugeTag {
            gauge,
            invariant,
            generator,
        } => format!(
            "GaugeTag(gauge={}, invariant={invariant}, generator={generator})",
            interner.resolve(*gauge)
        ),
        ax_ir::TensorProperty::HarmonicTag { basis, wave_symbol } => format!(
            "HarmonicTag(basis={}, wave_symbol={})",
            interner.resolve(*basis),
            wave_symbol
                .map(|sym| interner.resolve(sym).to_string())
                .unwrap_or_else(|| "None".to_string())
        ),
        ax_ir::TensorProperty::MatterTag(sym) => {
            format!("MatterTag({})", interner.resolve(*sym))
        }
        ax_ir::TensorProperty::TraceSpaceMeta(metadata) => format!(
            "TraceSpaceMeta(space_symbol={}, cyclic={})",
            interner.resolve(metadata.space_symbol),
            metadata.cyclic
        ),
    }
}

pub fn inspect_expr(expr: &Expr, env: &Env, interner: &ax_ir::Interner) -> InspectResult {
    let mut all_indices = Vec::new();
    collect_indices(expr, &mut all_indices);

    let mut by_name: HashMap<ax_ir::expr::Sym, Vec<ax_ir::Index>> = HashMap::new();
    for idx in all_indices {
        by_name.entry(idx.name).or_default().push(idx);
    }

    let mut free_indices = by_name
        .iter()
        .filter_map(|(name, occs)| {
            if occs.len() == 1 {
                let idx = &occs[0];
                Some((
                    interner.resolve(*name).to_string(),
                    match idx.variance {
                        ax_ir::Variance::Up => "up".to_string(),
                        ax_ir::Variance::Down => "down".to_string(),
                    },
                ))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    free_indices.sort();

    let mut dummy_pairs = by_name
        .iter()
        .filter_map(|(name, occs)| {
            if occs.len() == 2 && occs[0].variance != occs[1].variance {
                let rendered = interner.resolve(*name).to_string();
                Some((rendered.clone(), rendered))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    dummy_pairs.sort();

    let mut syms = Vec::new();
    collect_symbols(expr, &mut syms);
    syms.sort_by_key(|s| interner.resolve(*s).to_string());
    syms.dedup();

    let mut properties = syms
        .iter()
        .filter_map(|sym| {
            env.tensor_properties.get(sym).map(|props| {
                let mut rendered = props
                    .iter()
                    .map(|prop| property_name(prop, interner))
                    .collect::<Vec<_>>();
                rendered.sort();
                rendered.dedup();
                (interner.resolve(*sym).to_string(), rendered)
            })
        })
        .collect::<Vec<_>>();
    properties.sort_by(|a, b| a.0.cmp(&b.0));

    let symbols = syms
        .into_iter()
        .map(|sym| interner.resolve(sym).to_string())
        .collect::<Vec<_>>();

    InspectResult {
        kind: expr_kind(expr).to_string(),
        free_indices,
        dummy_pairs,
        properties,
        symbols,
        node_count: node_count(expr),
    }
}
