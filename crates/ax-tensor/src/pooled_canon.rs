use ax_ir::{ExprBuilder, ExprId, ExprPool, Interner, PooledExpr};

use crate::PropertyLookup;

pub fn canonicalise_pooled(
    id: ExprId,
    pool: &mut ExprPool,
    properties: &dyn PropertyLookup,
    interner: &Interner,
) -> ExprId {
    match pool.get(id).clone() {
        PooledExpr::Mul(_) => canonicalise_product_pooled(id, pool, properties, interner),
        PooledExpr::Add(terms) => {
            let canonical_terms = terms
                .into_iter()
                .map(|term| canonicalise_pooled(term, pool, properties, interner))
                .collect();
            ExprBuilder::new(pool).add(canonical_terms)
        }
        PooledExpr::Neg(inner) => {
            let canonical_inner = canonicalise_pooled(inner, pool, properties, interner);
            ExprBuilder::new(pool).neg(canonical_inner)
        }
        PooledExpr::Indexed(_, _) => {
            let product = pool.intern(PooledExpr::Mul(vec![id]));
            canonicalise_product_pooled(product, pool, properties, interner)
        }
        _ => id,
    }
}

fn canonicalise_product_pooled(
    id: ExprId,
    pool: &mut ExprPool,
    properties: &dyn PropertyLookup,
    interner: &Interner,
) -> ExprId {
    let expr = pool.to_expr(id);
    let result = crate::canonicalise(&expr, properties, interner);
    pool.from_expr(&result)
}

pub fn meld_pooled(
    id: ExprId,
    pool: &mut ExprPool,
    properties: &dyn PropertyLookup,
    interner: &Interner,
) -> ExprId {
    let expr = pool.to_expr(id);
    let result = crate::meld(&expr, properties, interner);
    pool.from_expr(&result)
}

pub fn simplify_pooled(
    id: ExprId,
    pool: &mut ExprPool,
    simplifier: &dyn Fn(&ax_ir::Expr, &Interner) -> ax_ir::Expr,
    interner: &Interner,
) -> ExprId {
    let expr = pool.to_expr(id);
    let result = simplifier(&expr, interner);
    pool.from_expr(&result)
}

pub fn collect_terms_pooled(id: ExprId, pool: &mut ExprPool, interner: &Interner) -> ExprId {
    let expr = pool.to_expr(id);
    let result = crate::collect_terms_expr(&expr, interner);
    pool.from_expr(&result)
}
