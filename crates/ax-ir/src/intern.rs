pub struct Interner {
    rodeo: lasso::ThreadedRodeo,
}

impl Interner {
    pub fn new() -> Self {
        Self {
            rodeo: lasso::ThreadedRodeo::new(),
        }
    }

    pub fn get_or_intern(&self, s: &str) -> lasso::Spur {
        self.rodeo.get_or_intern(s)
    }

    pub fn resolve(&self, key: lasso::Spur) -> &str {
        self.rodeo.resolve(&key)
    }
}

impl Default for Interner {
    fn default() -> Self {
        Self::new()
    }
}
