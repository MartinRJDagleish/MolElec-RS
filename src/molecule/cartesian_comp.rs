use strum_macros::EnumIter;
use strum::IntoEnumIterator;

#[repr(usize)]
#[derive(Clone, Copy, Debug, EnumIter, PartialEq)] // Add PartialEq trait
pub(crate) enum Cartesian {
    X = 0usize,
    Y = 1usize,
    Z = 2usize,
}

pub(crate) const CC_X: usize = Cartesian::X as usize;
pub(crate) const CC_Y: usize = Cartesian::Y as usize;
pub(crate) const CC_Z: usize = Cartesian::Z as usize;

mod tests {
    #[test]
    fn test_cartesian() {
        use super::*;
        let cart = Cartesian::iter().collect::<Vec<_>>();
        assert_eq!(cart, vec![Cartesian::X, Cartesian::Y, Cartesian::Z]);
    }
}
