use ax_ir::Expr;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_bigint::BigInt;
use num_rational::BigRational;

fn ket(index: usize, dim: usize) -> Vec<Expr> {
    (0..dim)
        .map(|i| {
            if i == index {
                Expr::one()
            } else {
                Expr::zero()
            }
        })
        .collect()
}

fn bell_pair_density() -> Vec<Vec<Expr>> {
    let half = Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2)));
    let mut rho = vec![vec![Expr::zero(); 4]; 4];
    rho[0][0] = half.clone();
    rho[0][3] = half.clone();
    rho[3][0] = half.clone();
    rho[3][3] = half;
    rho
}

fn basis_density(index: usize, dim: usize) -> Vec<Vec<Expr>> {
    ax_qm::density_matrix(&ket(index, dim))
}

fn bench_partial_trace(c: &mut Criterion) {
    let interner = ax_ir::Interner::new();
    let bell = bell_pair_density();
    c.bench_function("partial_trace_bell_2x2_trace_b", |b| {
        b.iter(|| ax_qm::partial_trace(black_box(&bell), 2, 2, 'B', black_box(&interner)))
    });

    let three_qubit_density = basis_density(5, 8);
    c.bench_function("partial_trace_three_qubit_2x4_trace_b", |b| {
        b.iter(|| {
            ax_qm::partial_trace(
                black_box(&three_qubit_density),
                2,
                4,
                'B',
                black_box(&interner),
            )
        })
    });
}

criterion_group!(benches, bench_partial_trace);
criterion_main!(benches);
