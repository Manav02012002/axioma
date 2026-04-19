use rowan::SyntaxKind as RowanKind;

#[repr(u16)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SyntaxKind {
    Eof = 0,

    Ident,
    Int,
    Float,
    String,

    KwModule,
    KwImport,
    KwLet,
    KwIn,
    KwIndexset,
    KwTrue,
    KwFalse,

    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    Dagger,
    TensorProduct,

    Eq,
    Less,
    Greater,
    Pipe,
    Comma,
    Dot,
    Colon,
    Semi,

    LParen,
    RParen,
    LBrace,
    RBrace,
    LBrack,
    RBrack,

    Underscore,

    Whitespace,
    CommentLine,
    CommentBlock,

    Error,

    ExprStmt,
    CallExpr,
    ListExpr,
    KetExpr,
    BraExpr,
    BraKetExpr,
    DaggerExpr,
    TensorProductExpr,
    NamedArg,
    StringLiteral,
    BoolLiteral,
    TableauSymmetryExpr,
    TableauShapeList,
    TableauSlotMapList,
    TableauLabels,
    TableauTraceFreeList,

    Root,
}

impl From<SyntaxKind> for RowanKind {
    fn from(k: SyntaxKind) -> Self {
        RowanKind(k as u16)
    }
}

pub struct T;

impl T {
    pub const EOF: SyntaxKind = SyntaxKind::Eof;
    pub const IDENT: SyntaxKind = SyntaxKind::Ident;
    pub const INT: SyntaxKind = SyntaxKind::Int;
    pub const FLOAT: SyntaxKind = SyntaxKind::Float;
    pub const STRING: SyntaxKind = SyntaxKind::String;
    pub const WHITESPACE: SyntaxKind = SyntaxKind::Whitespace;
    pub const COMMENT_LINE: SyntaxKind = SyntaxKind::CommentLine;
    pub const COMMENT_BLOCK: SyntaxKind = SyntaxKind::CommentBlock;
    pub const ERROR: SyntaxKind = SyntaxKind::Error;
    pub const ROOT: SyntaxKind = SyntaxKind::Root;
}

pub fn is_trivia(k: SyntaxKind) -> bool {
    matches!(
        k,
        SyntaxKind::Whitespace | SyntaxKind::CommentLine | SyntaxKind::CommentBlock
    )
}
