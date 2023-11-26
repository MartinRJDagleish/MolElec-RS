use crate::molecule::cartesian_comp::{CC_X, CC_Y, CC_Z};
use ndarray::{Array1, ArrayView1};
use physical_constants::ALPHA_PARTICLE_ELECTRON_MASS_RATIO;

#[derive(Debug, Default)]
struct E_herm_coeff_3d {
    // Coefficients for the Hermite expansion of Cartesian Gaussian functions
    // Generalized to work for normal ints AND derivatives
    // See: Molecular Electronic-Structure Theory, Helgaker, Jorgensen, Olsen, 2000,
    E_ij: E_herm_coeff_1d, // x comp
    E_kl: E_herm_coeff_1d, // y comp
    E_mn: E_herm_coeff_1d, // z comp
}

#[derive(Debug, Default)]
struct E_herm_coeff_1d {
    // Coefficients for the Hermite expansion of Cartesian Gaussian functions (1d)
    // See: Molecular Electronic-Structure Theory, Helgaker, Jorgensen, Olsen, 2000,
    alpha1: f64,
    alpha2: f64,
    one_over_alph_p: f64,
    dist_AB_comp: f64, // x, y, or z component of the distance between A and B
}

#[derive(Debug, Default)]
struct R_herm_aux_int {
    // Hermite auxiliary int for the Hermite expansion of Cartesian Gaussian functions
    // See: Molecular Electronic-Structure Theory, Helgaker, Jorgensen, Olsen, 2000,
    boys_val: f64,
    vec_PC: Array1<f64>, //TODO: change this?
    alph_p: f64,
}

impl From<(E_herm_coeff_1d, E_herm_coeff_1d, E_herm_coeff_1d)> for E_herm_coeff_3d {
    fn from((E_ij, E_kl, E_mn): (E_herm_coeff_1d, E_herm_coeff_1d, E_herm_coeff_1d)) -> Self {
        Self { E_ij, E_kl, E_mn }
    }
}

impl E_herm_coeff_3d {
    /// ### Note: 
    /// `vec_BA` is the vector from B to A, i.e. A - B (not B - A) => BA_x = A_x - B_x
    /// 
    /// ### Arguments
    /// ----------
    /// `alpha1` : Exponent of the first Gaussian function.
    /// 
    /// `alpha2` : Exponent of the second Gaussian function.
    /// 
    /// `vec_BA` : Vector from B to A, i.e. A - B (not B - A) => BA_x = A_x - B_x
    /// 
    /// ### Returns
    /// ----------
    /// `Self` : Struct containing the coefficients for the Hermite expansion of Cartesian Gaussian functions
    fn new(alpha1: f64, alpha2: f64, vec_BA: ArrayView1<f64>) -> Self {
        let one_over_alph_p = 1.0/ (alpha1 + alpha2);
        let E_ij = E_herm_coeff_1d::new(alpha1, alpha2, one_over_alph_p, vec_BA[CC_X]);
        let E_kl = E_herm_coeff_1d::new(alpha1, alpha2, one_over_alph_p, vec_BA[CC_Y]);
        let E_mn = E_herm_coeff_1d::new(alpha1, alpha2, one_over_alph_p, vec_BA[CC_Z]);

        Self { E_ij, E_kl, E_mn }
    }

    fn calc_recurr_rel(&mut self, l1: i32, l2: i32, no_nodes: i32, deriv_deg: i32) -> f64 {
        let E_ij_val = self.E_ij.calc_recurr_rel(l1, l2, no_nodes, deriv_deg);
        let E_kl_val = self.E_kl.calc_recurr_rel(l1, l2, no_nodes, deriv_deg);
        let E_mn_val = self.E_mn.calc_recurr_rel(l1, l2, no_nodes, deriv_deg);
        E_ij_val * E_kl_val * E_mn_val // return product of all three components
    }
}

impl E_herm_coeff_1d {
    fn new(alpha1: f64, alpha2: f64, one_over_alph_p: f64, dist_AB_comp: f64) -> Self {
        Self {
            alpha1,
            alpha2,
            one_over_alph_p,
            dist_AB_comp,
        }
    }

    //[ ] make this use the structs (E_herm_coeff_1d and E_herm_coeff_3d) instead of the arguments
    /// Calculate the expansion coefficient for the overlap integral
    /// between two contracted Gaussian functions.
    ///
    /// ### Arguments
    /// ----------
    /// `l1` : Cartesian angular momentum of the first Gaussian function. (for x, y, or z)
    ///
    /// `l2` : Cartesian angular momentum of the second Gaussian function. (for x, y, or z)
    ///
    /// `no_nodes` : Number of nodes in Hermite (depends on type of int, e.g. always zero for overlap).
    ///
    /// `deriv_deg` : Degree of derivative (0 for overlap and mol. ints, 1 for first derivative, etc.)
    #[inline]
    pub fn calc_recurr_rel(&self, l1: i32, l2: i32, no_nodes: i32, deriv_deg: i32) -> f64 {
        // Early return
        if no_nodes < 0 || no_nodes > (l1 + l2) || deriv_deg < 0 {
            return 0.0;
        }
        let one_over_2p = 0.5 * self.one_over_alph_p;
        let mu = self.alpha1 * self.alpha2 * self.one_over_alph_p;
        let q = -2.0 * mu;

        match (no_nodes, l1, l2, deriv_deg) {
            // Molecular integral cases; works and is correct
            (0, 0, 0, 0) => (-mu * self.dist_AB_comp * self.dist_AB_comp).exp(),
            (_, _, 0, 0) => {
                //* decrement index l1
                one_over_2p * self.calc_recurr_rel(l1 - 1, l2, no_nodes - 1, deriv_deg)
                    - (self.alpha2 * self.one_over_alph_p * self.dist_AB_comp)
                        * self.calc_recurr_rel(l1 - 1, l2, no_nodes, deriv_deg)
                    + (no_nodes + 1) as f64
                        * self.calc_recurr_rel(l1 - 1, l2, no_nodes + 1, deriv_deg)
            }
            (_, _, _, 0) => {
                //* decrement index l2
                one_over_2p * self.calc_recurr_rel(l1, l2 - 1, no_nodes - 1, deriv_deg)
                    + (self.alpha1 * self.one_over_alph_p * self.dist_AB_comp)
                        * self.calc_recurr_rel(l1, l2 - 1, no_nodes, deriv_deg)
                    + (no_nodes + 1) as f64
                        * self.calc_recurr_rel(l1, l2 - 1, no_nodes + 1, deriv_deg)
            }
            // Derivate cases
            (_,_,0,_) => todo!(), 
            _ => todo!()
            // [ ] implement the rest of the cases -> mainly derivative cases
        }
    }
}

impl R_herm_aux_int {
    // Use Boys function to calculate the Hermite auxiliary integral
    fn new() {
        todo!();
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn test_E_calc_recurr_rel() {
        let test_vec_AB = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let mut E_ab = E_herm_coeff_3d::new(0.5, 0.5, ArrayView1::from(&test_vec_AB));

        let l1 = 2;
        let l2 = 1;
        let no_nodes = 0;
        let deriv_deg = 0;
        
        let result = E_ab.calc_recurr_rel(l1, l2, no_nodes, deriv_deg);
        println!("result: {}", result);
        assert_abs_diff_eq!(result, -0.0049542582177241, epsilon = 1e-10);
    }
}
