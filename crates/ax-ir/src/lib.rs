pub mod expr;
pub mod intern;
pub mod pool;
pub mod pretty;
pub mod symmetry;

use std::cell::Cell;
use std::cell::RefCell;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Instant;

pub use expr::{
    Assumption, Condition, Convention, Expr, FourierSign, Grading, Index, IndexFamily,
    IndexPosition, LeviCivitaNorm, MetricSignature, ParentRel, RicciContraction, RiemannSign,
    TensorProperty, TrustLevel, Variance,
};
pub use intern::Interner;
pub use pool::{ExprBuilder, ExprId, ExprPool, PooledExpr};
pub use pretty::pretty_print;
pub use symmetry::{
    validate_duality_in_dimension, validate_tableau_attachment, validate_tensor_symmetry,
    DimensionGuard, DualityKind, DualityValidationError, RestrictedSymmetryMode,
    SymmetrySource, SymmetryValidationError, TableauAttachment, TensorSymmetry,
};

thread_local! {
    static CURRENT_DEADLINE: Cell<Option<Instant>> = const { Cell::new(None) };
    static CURRENT_CANCELLATION: RefCell<Option<CancellationToken>> = const { RefCell::new(None) };
}

#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionAbort {
    Interrupted,
    TimedOut,
}

pub fn current_deadline() -> Option<Instant> {
    CURRENT_DEADLINE.with(Cell::get)
}

pub fn with_deadline<T>(deadline: Option<Instant>, f: impl FnOnce() -> T) -> T {
    CURRENT_DEADLINE.with(|slot| {
        let previous = slot.replace(deadline);
        let out = f();
        slot.set(previous);
        out
    })
}

pub fn with_cancellation<T>(token: Option<CancellationToken>, f: impl FnOnce() -> T) -> T {
    CURRENT_CANCELLATION.with(|slot| {
        let previous = slot.replace(token);
        let out = f();
        let _ = slot.replace(previous);
        out
    })
}

pub fn current_cancellation() -> Option<CancellationToken> {
    CURRENT_CANCELLATION.with(|slot| slot.borrow().clone())
}

pub fn check_deadline() -> Result<(), String> {
    if current_cancellation()
        .as_ref()
        .is_some_and(CancellationToken::is_cancelled)
    {
        return Err("execution interrupted".to_string());
    }
    if let Some(deadline) = current_deadline() {
        if Instant::now() > deadline {
            return Err("computation timed out".to_string());
        }
    }
    Ok(())
}

pub fn abort_if_cancelled() {
    match check_deadline() {
        Ok(()) => {}
        Err(message) if message == "execution interrupted" => {
            std::panic::panic_any(ExecutionAbort::Interrupted)
        }
        Err(_) => std::panic::panic_any(ExecutionAbort::TimedOut),
    }
}

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
