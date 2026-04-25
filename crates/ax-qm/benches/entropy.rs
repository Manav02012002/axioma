use ax_ir::Expr;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_bigint::BigInt;
use num_rational::BigRational;

fn rational(numer: i64, denom: i64) -> Expr {
    Expr::Rational(BigRational::new(BigInt::from(numer), BigInt::from(denom)))
}

fn diagonal_density(entries: Vec<Expr>) -> Vec<Vec<Expr>> {
    let dim = entries.len();
    let mut rho = vec![vec![Expr::zero(); dim]; dim];
    for (idx, entry) in entries.into_iter().enumerate() {
        rho[idx][idx] = entry;
    }
    rho
}

fn bench_entropy(c: &mut Criterion) {
    let interner = ax_ir::Interner::new();
    let qubit_mixed = diagonal_density(vec![rational(1, 2), rational(1, 2)]);
    c.bench_function("von_neumann_entropy_diagonal_qubit_2x2", |b| {
        b.iter(|| ax_qm::von_neumann_entropy(black_box(&qubit_mixed), black_box(&interner)))
    });

    let qutrit_mixed = diagonal_density(vec![rational(1, 2), rational(1, 3), rational(1, 6)]);
    c.bench_function("von_neumann_entropy_supported_diagonal_qutrit_3x3", |b| {
        b.iter(|| ax_qm::von_neumann_entropy(black_box(&qutrit_mixed), black_box(&interner)))
    });
}

criterion_group!(benches, bench_entropy);
criterion_main!(benches);
