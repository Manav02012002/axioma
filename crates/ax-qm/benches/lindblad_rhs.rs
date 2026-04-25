use ax_ir::Expr;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn zero_matrix(dim: usize) -> Vec<Vec<Expr>> {
    vec![vec![Expr::zero(); dim]; dim]
}

fn basis_density(index: usize, dim: usize) -> Vec<Vec<Expr>> {
    let mut rho = zero_matrix(dim);
    rho[index][index] = Expr::one();
    rho
}

fn transition_jump(to: usize, from: usize, dim: usize) -> Vec<Vec<Expr>> {
    let mut jump = zero_matrix(dim);
    jump[to][from] = Expr::one();
    jump
}

fn bench_lindblad_rhs(c: &mut Criterion) {
    let interner = ax_ir::Interner::new();
    let h_qubit = zero_matrix(2);
    let rho_qubit = basis_density(1, 2);
    let jumps_qubit = vec![transition_jump(0, 1, 2)];
    c.bench_function("lindblad_rhs_amplitude_damping_qubit_2x2", |b| {
        b.iter(|| {
            ax_qm::lindblad_rhs(
                black_box(&h_qubit),
                black_box(&rho_qubit),
                black_box(&jumps_qubit),
                black_box(&interner),
            )
        })
    });

    let h_medium = zero_matrix(8);
    let rho_medium = basis_density(7, 8);
    let jumps_medium = (0..7)
        .map(|from| transition_jump(from, from + 1, 8))
        .collect::<Vec<_>>();
    c.bench_function("lindblad_rhs_decay_chain_8x8", |b| {
        b.iter(|| {
            ax_qm::lindblad_rhs(
                black_box(&h_medium),
                black_box(&rho_medium),
                black_box(&jumps_medium),
                black_box(&interner),
            )
        })
    });
}

criterion_group!(benches, bench_lindblad_rhs);
criterion_main!(benches);
