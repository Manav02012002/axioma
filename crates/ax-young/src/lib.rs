#![forbid(unsafe_code)]
//! Representation-theory kernel for Young diagrams, tableaux, Littlewood-Richardson products,
//! hook/content dimensions, Garnir standardization, lazy projector descriptions,
//! and group-backed projector actions.

pub mod dimension_identities;
mod dimension;
mod error;
mod garnir;
pub mod group_action;
mod branching;
pub mod duality;
mod lr;
mod plethysm;
mod partition;
mod projector;
mod rep_ring;
mod render;
mod schur;
mod skew;
mod semistandard;
mod ssyt_enum;
mod tableau;
mod lr_basis;

pub use dimension::{
    dimension_gl, dimension_gl_u64_saturating, dimension_of_rep_expansion,
    dimension_of_representation, dimension_of_schur_expansion, hook_content_factors,
};
pub use dimension_identities::{
    first_nonvanishing_gl_dimension, schouten_annihilates_antisym_degree,
    tableau_requires_dimension_annihilation, tensor_symmetry_annihilates_in_dimension,
    vanishes_in_gl_dimension,
};
pub use duality::{
    hodge_dual_form_degree, induced_form_tableau_duality, is_middle_degree,
    selfdual_eigenspace_dimension,
};
pub use error::YoungError;
pub use branching::{branch_gl_n_to_gl_n_minus_1, branch_s_n_to_s_n_minus_1};
pub use garnir::standardize_garnir;
pub use group_action::{
    build_group_backed_projector, build_projector_with_trace, build_stabilizer_group,
    canonicalize_slots_under_both_groups, canonicalize_slots_under_row_group,
    canonicalize_slots_with_trace, column_group_orbits, expand_projector_group_algebra,
    row_group_orbits, validate_perm, GroupBackedProjector, GroupProjectorError,
    ProjectorNormalization, StabilizerGroup,
};
pub use lr::{lr_shapes, lr_tensor};
pub use lr_basis::{
    littlewood_richardson_basis, littlewood_richardson_coefficient,
    lr_shapes_with_multiplicity, LittlewoodRichardsonBasisEntry,
};
pub use partition::YoungDiagram;
pub use plethysm::{plethysm_rep_expansion, plethysm_schur_by_shape};
pub use projector::{
    column_antisymmetrizer_generators, expand_lazy_projector, lazy_projector,
    row_symmetrizer_generators, LazyProjector, PermutationTerm, Symmetriser, SymmetryOperatorKind,
};
pub use rep_ring::{
    add_rep_expansions, multiply_rep_expansions, tensor_product_decomposition, MultiplicitySpace,
    RepExpansion, SchurExpansion, TensorProductDecomposition,
};
pub use schur::{schur_basis_shape, schur_tensor_product};
pub use semistandard::{kostka_number, SemistandardTableau};
pub use skew::SkewDiagram;
pub use ssyt_enum::{
    enumerate_semistandard_with_content, enumerate_skew_semistandard_with_content,
    kostka_number_exact,
};
pub use tableau::{FilledTableau, Tableaux, YoungTableau};
