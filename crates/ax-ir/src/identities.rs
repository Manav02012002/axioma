#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorMultitermIdentity {
    FirstBianchi { cyclic_slots: [usize; 3] },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorIdentitySet {
    pub multiterm: Vec<TensorMultitermIdentity>,
}

impl TensorIdentitySet {
    pub fn empty() -> Self {
        Self {
            multiterm: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.multiterm.is_empty()
    }
}

pub fn riemann_identity_set() -> TensorIdentitySet {
    TensorIdentitySet {
        multiterm: vec![TensorMultitermIdentity::FirstBianchi {
            cyclic_slots: [1, 2, 3],
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn riemann_identity_set_uses_exact_slot_convention() {
        assert_eq!(
            riemann_identity_set(),
            TensorIdentitySet {
                multiterm: vec![TensorMultitermIdentity::FirstBianchi {
                    cyclic_slots: [1, 2, 3]
                }],
            }
        );
    }
}
