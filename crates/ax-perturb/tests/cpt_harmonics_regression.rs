use ax_ir::Interner;
use ax_perturb::{
    derive_linear_tensor_einstein_equations, derive_linear_vector_einstein_equations_poisson,
    gauge::svt_decompose_perturbation, project_scalar_equations_to_harmonic_space,
    project_tensor_equations_to_harmonic_space, project_vector_equations_to_harmonic_space,
    scalar_laplacian_eigenvalue, tensor_laplacian_eigenvalue, vector_laplacian_eigenvalue,
    FrwBackgroundSpec, SpatialCurvature,
};

#[test]
fn scalar_harmonic_eigenvalues_are_different_across_curvatures() {
    let interner = Interner::new();
    let flat = scalar_laplacian_eigenvalue(SpatialCurvature::Flat, &interner).unwrap();
    let closed = scalar_laplacian_eigenvalue(SpatialCurvature::Closed, &interner).unwrap();
    let open = scalar_laplacian_eigenvalue(SpatialCurvature::Open, &interner).unwrap();

    assert_ne!(flat, closed);
    assert_ne!(flat, open);
    assert_ne!(closed, open);
}

#[test]
fn vector_harmonic_eigenvalues_are_different_across_curvatures() {
    let interner = Interner::new();
    let flat = vector_laplacian_eigenvalue(SpatialCurvature::Flat, &interner).unwrap();
    let closed = vector_laplacian_eigenvalue(SpatialCurvature::Closed, &interner).unwrap();
    let open = vector_laplacian_eigenvalue(SpatialCurvature::Open, &interner).unwrap();

    assert_ne!(flat, closed);
    assert_ne!(flat, open);
    assert_ne!(closed, open);
}

#[test]
fn tensor_harmonic_eigenvalues_are_different_across_curvatures() {
    let interner = Interner::new();
    let flat = tensor_laplacian_eigenvalue(SpatialCurvature::Flat, &interner).unwrap();
    let closed = tensor_laplacian_eigenvalue(SpatialCurvature::Closed, &interner).unwrap();
    let open = tensor_laplacian_eigenvalue(SpatialCurvature::Open, &interner).unwrap();

    assert_ne!(flat, closed);
    assert_ne!(flat, open);
    assert_ne!(closed, open);
}

#[test]
fn harmonic_projection_preserves_equation_labels() {
    let interner = Interner::new();
    let bg = FrwBackgroundSpec::default_flat_conformal(&interner);

    let decomp = svt_decompose_perturbation(3, &interner).unwrap();
    let scalar =
        ax_perturb::cosmology::linearized_einstein_scalar(&bg, &decomp, &interner).unwrap();
    let scalar_projected =
        project_scalar_equations_to_harmonic_space(&scalar, &bg, &interner).unwrap();
    assert_eq!(
        scalar.iter().map(|eq| eq.label.clone()).collect::<Vec<_>>(),
        scalar_projected
            .equations
            .iter()
            .map(|eq| eq.label.clone())
            .collect::<Vec<_>>()
    );

    let vector = derive_linear_vector_einstein_equations_poisson(&bg, &interner)
        .unwrap()
        .equations;
    let vector_projected =
        project_vector_equations_to_harmonic_space(&vector, &bg, &interner).unwrap();
    assert_eq!(
        vector.iter().map(|eq| eq.label.clone()).collect::<Vec<_>>(),
        vector_projected
            .equations
            .iter()
            .map(|eq| eq.label.clone())
            .collect::<Vec<_>>()
    );

    let tensor = derive_linear_tensor_einstein_equations(&bg, &interner)
        .unwrap()
        .equations;
    let tensor_projected =
        project_tensor_equations_to_harmonic_space(&tensor, &bg, &interner).unwrap();
    assert_eq!(
        tensor.iter().map(|eq| eq.label.clone()).collect::<Vec<_>>(),
        tensor_projected
            .equations
            .iter()
            .map(|eq| eq.label.clone())
            .collect::<Vec<_>>()
    );
}
