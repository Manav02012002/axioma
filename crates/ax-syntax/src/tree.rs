use rowan::{ast::AstNode, GreenNode, Language, TextSize};

use crate::kind::SyntaxKind;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AxLanguage {}

impl Language for AxLanguage {
    type Kind = SyntaxKind;

    fn kind_from_raw(raw: rowan::SyntaxKind) -> SyntaxKind {
        unsafe { std::mem::transmute::<u16, SyntaxKind>(raw.0) }
    }

    fn kind_to_raw(kind: SyntaxKind) -> rowan::SyntaxKind {
        rowan::SyntaxKind(kind as u16)
    }
}

pub type SyntaxNode = rowan::SyntaxNode<AxLanguage>;
pub type SyntaxToken = rowan::SyntaxToken<AxLanguage>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableauSymmetryExpr {
    syntax: SyntaxNode,
}

impl AstNode for TableauSymmetryExpr {
    type Language = AxLanguage;

    fn can_cast(kind: SyntaxKind) -> bool {
        kind == SyntaxKind::TableauSymmetryExpr
    }

    fn cast(syntax: SyntaxNode) -> Option<Self> {
        Self::can_cast(syntax.kind()).then_some(Self { syntax })
    }

    fn syntax(&self) -> &SyntaxNode {
        &self.syntax
    }
}

impl TableauSymmetryExpr {
    /// Return the parsed tableau shape list.
    pub fn tableau_shapes(&self) -> Vec<Vec<usize>> {
        self.child_numbers(SyntaxKind::TableauShapeList)
    }

    /// Return the parsed tableau slot-map list.
    pub fn tableau_slot_maps(&self) -> Vec<Vec<usize>> {
        self.child_numbers(SyntaxKind::TableauSlotMapList)
    }

    /// Return the parsed tableau labels, defaulting to `None` when absent.
    pub fn tableau_labels(&self) -> Vec<Option<String>> {
        let shapes = self.tableau_shapes();
        match self.child_strings(SyntaxKind::TableauLabels) {
            Some(mut labels) => {
                labels.resize(shapes.len(), String::new());
                labels
                    .into_iter()
                    .take(shapes.len())
                    .map(|label| if label.is_empty() { None } else { Some(label) })
                    .collect()
            }
            None => vec![None; shapes.len()],
        }
    }

    /// Return the parsed trace-free flags, defaulting to `false` when absent.
    pub fn tableau_trace_free_flags(&self) -> Vec<bool> {
        let shapes = self.tableau_shapes();
        match self.child_bools(SyntaxKind::TableauTraceFreeList) {
            Some(mut flags) => {
                flags.resize(shapes.len(), false);
                flags.into_iter().take(shapes.len()).collect()
            }
            None => vec![false; shapes.len()],
        }
    }

    pub fn lower_tensor_symmetry(&self) -> Option<ax_ir::TensorSymmetry> {
        let shapes = self.tableau_shapes();
        let slots = self.tableau_slot_maps();
        if shapes.is_empty() || shapes.len() != slots.len() {
            return None;
        }

        let labels = self.tableau_labels();
        if labels.len() != shapes.len() {
            return None;
        }

        let trace_free = self.tableau_trace_free_flags();
        if trace_free.len() != shapes.len() {
            return None;
        }

        let tableaux = shapes
            .into_iter()
            .zip(slots)
            .zip(labels)
            .zip(trace_free)
            .map(
                |(((shape, slot_map), label), trace_free)| ax_ir::TableauAttachment {
                    shape,
                    slot_map,
                    multiplicity_numer: 1,
                    multiplicity_denom: 1,
                    duality: ax_ir::DualityKind::None,
                    restricted_mode: ax_ir::RestrictedSymmetryMode::FullYoung,
                    trace_free,
                    dimension_guard: None,
                    source: ax_ir::SymmetrySource::Declared,
                    label,
                },
            )
            .collect::<Vec<_>>();

        Some(ax_ir::TensorSymmetry {
            tableaux,
            inherits_under_derivative: false,
            inherits_under_tensor_product: false,
            inherits_under_contraction: false,
            preserves_trace_free_under_projection: false,
        })
    }

    fn child_numbers(&self, kind: SyntaxKind) -> Vec<Vec<usize>> {
        self.syntax
            .children()
            .find(|child| child.kind() == kind)
            .map(parse_nested_usize_values)
            .unwrap_or_default()
    }

    fn child_strings(&self, kind: SyntaxKind) -> Option<Vec<String>> {
        self.syntax
            .children()
            .find(|child| child.kind() == kind)
            .map(parse_string_values)
    }

    fn child_bools(&self, kind: SyntaxKind) -> Option<Vec<bool>> {
        self.syntax
            .children()
            .find(|child| child.kind() == kind)
            .map(parse_bool_values)
    }
}

pub fn syntax_node_from_green(green: GreenNode) -> SyntaxNode {
    SyntaxNode::new_root(green)
}

pub fn tableau_symmetry_exprs(root: &SyntaxNode) -> Vec<TableauSymmetryExpr> {
    root.descendants()
        .filter_map(TableauSymmetryExpr::cast)
        .collect()
}

pub fn tableau_symmetry_expr_at_offset(
    root: &SyntaxNode,
    offset: usize,
) -> Option<TableauSymmetryExpr> {
    let offset = TextSize::from(offset as u32);
    root.descendants()
        .filter_map(TableauSymmetryExpr::cast)
        .find(|expr| {
            let range = expr.syntax().text_range();
            range.start() <= offset && offset <= range.end()
        })
}

fn parse_nested_usize_values(node: SyntaxNode) -> Vec<Vec<usize>> {
    let mut nested = Vec::new();
    let child_lists = node
        .children()
        .filter(|child| child.kind() == SyntaxKind::ListExpr)
        .collect::<Vec<_>>();
    if child_lists.is_empty() {
        return vec![parse_usize_tokens(&node)];
    }
    for child in child_lists {
        nested.push(parse_usize_tokens(&child));
    }
    nested
}

fn parse_usize_tokens(node: &SyntaxNode) -> Vec<usize> {
    node.descendants_with_tokens()
        .filter_map(|element| element.into_token())
        .filter(|token| token.kind() == SyntaxKind::Int)
        .filter_map(|token| token.text().parse::<usize>().ok())
        .collect()
}

fn parse_string_values(node: SyntaxNode) -> Vec<String> {
    node.descendants_with_tokens()
        .filter_map(|element| element.into_token())
        .filter(|token| token.kind() == SyntaxKind::String)
        .map(|token| {
            token
                .text()
                .strip_prefix('"')
                .and_then(|value| value.strip_suffix('"'))
                .unwrap_or(token.text())
                .to_string()
        })
        .collect()
}

fn parse_bool_values(node: SyntaxNode) -> Vec<bool> {
    node.descendants_with_tokens()
        .filter_map(|element| element.into_token())
        .filter_map(|token| match token.kind() {
            SyntaxKind::KwTrue => Some(true),
            SyntaxKind::KwFalse => Some(false),
            _ => None,
        })
        .collect()
}
