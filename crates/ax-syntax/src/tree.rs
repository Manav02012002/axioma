use rowan::{GreenNode, Language};

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

pub fn syntax_node_from_green(green: GreenNode) -> SyntaxNode {
    SyntaxNode::new_root(green)
}
