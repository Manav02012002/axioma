use ax_ir::Expr;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn zero_matrix(dim: usize) -> Vec<Vec<Expr>> {
    vec![vec![Expr::zero(); dim]; dim]
}

fn transition_jump(to: usize, from: usize, dim: usize) -> Vec<Vec<Expr>> {
    let mut jump = zero_matrix(dim);
    jump[to][from] = Expr::one();
    jump
}

fn bench_steady_state(c: &mut Criterion) {
    let interner = ax_ir::Interner::new();
    let h_qubit = zero_matrix(2);
    let jumps_qubit = vec![transition_jump(0, 1, 2)];
    c.bench_function("steady_state_amplitude_damping_qubit_2x2", |b| {
        b.iter(|| {
            ax_solve::lindblad_steady_state_linear(
                black_box(&h_qubit),
                black_box(&jumps_qubit),
                black_box(&interner),
            )
        })
    });

    let h_medium = zero_matrix(4);
    let jumps_medium = (0..3)
        .map(|from| transition_jump(from, from + 1, 4))
        .collect::<Vec<_>>();
    c.bench_function("steady_state_decay_chain_4x4", |b| {
        b.iter(|| {
            ax_solve::lindblad_steady_state_linear(
                black_box(&h_medium),
                black_box(&jumps_medium),
                black_box(&interner),
            )
        })
    });
}

criterion_group!(benches, bench_steady_state);
criterion_main!(benches);
