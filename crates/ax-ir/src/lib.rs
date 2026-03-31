pub mod expr;
pub mod intern;
pub mod pretty;

pub use expr::{
    Assumption, Condition, Convention, Expr, FourierSign, Grading, Index, IndexFamily,
    IndexPosition, LeviCivitaNorm, MetricSignature, RicciContraction, RiemannSign,
    TensorProperty, TrustLevel, Variance,
};
pub use intern::Interner;
pub use pretty::pretty_print;

#[cfg(test)]
mod tests {
    use super::*;
    use lasso::Key;

    #[test]
    fn add_flattens_and_combines() {
        let a = Expr::add(vec![
            Expr::Int(1.into()),
            Expr::Int(2.into()),
            Expr::Int(3.into()),
        ]);
        assert_eq!(a, Expr::Int(6.into()));
    }

    #[test]
    fn add_drops_zeros() {
        let s = Expr::Sym(lasso::Spur::try_from_usize(0).unwrap());
        let a = Expr::add(vec![Expr::zero(), s.clone()]);
        assert_eq!(a, s);
    }

    #[test]
    fn mul_zero_annihilates() {
        let s = Expr::Sym(lasso::Spur::try_from_usize(0).unwrap());
        let m = Expr::mul(vec![Expr::zero(), s]);
        assert_eq!(m, Expr::zero());
    }

    #[test]
    fn mul_combines_numerics() {
        let m = Expr::mul(vec![Expr::Int(3.into()), Expr::Int(4.into())]);
        assert_eq!(m, Expr::Int(12.into()));
    }

    #[test]
    fn pow_simplifies() {
        let s = Expr::Sym(lasso::Spur::try_from_usize(0).unwrap());
        assert_eq!(Expr::pow(s.clone(), Expr::zero()), Expr::one());
        assert_eq!(Expr::pow(s.clone(), Expr::one()), s);
    }

    #[test]
    fn neg_double_cancels() {
        let s = Expr::Sym(lasso::Spur::try_from_usize(0).unwrap());
        let nn = Expr::neg(Expr::neg(s.clone()));
        assert_eq!(nn, s);
    }
}
