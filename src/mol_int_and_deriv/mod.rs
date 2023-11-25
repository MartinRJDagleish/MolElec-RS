use crate::molecule::cartesian_comp::{CC_X, CC_Y, CC_Z};
use ndarray::{Array1, ArrayView1};

#[derive(Debug, Default)]
struct E_herm_coeff_3d {
    // Coefficients for the Hermite expansion of Cartesian Gaussian functions
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
    alph_p: f64,
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
    fn new(alpha1: f64, alpha2: f64, alph_p: f64, vec_AB: ArrayView1<f64>) -> Self {
        let E_ij = E_herm_coeff_1d::new(alpha1, alpha2, alph_p, vec_AB[CC_X]);
        let E_kl = E_herm_coeff_1d::new(alpha1, alpha2, alph_p, vec_AB[CC_Y]);
        let E_mn = E_herm_coeff_1d::new(alpha1, alpha2, alph_p, vec_AB[CC_Z]);

        Self { E_ij, E_kl, E_mn }
    }
    
    // fn calc_recurr_rel(&mut self) {
    //
    // }
}

impl E_herm_coeff_1d {
    fn new(alpha1: f64, alpha2: f64, alph_p: f64, dist_AB_comp: f64) -> Self {
        Self {
            alpha1,
            alpha2,
            alph_p,
            dist_AB_comp,
        }
    }

    //TODO: make this use the structs (E_herm_coeff_1d and E_herm_coeff_3d) instead of the arguments
    #[inline]
    pub fn calc_recurr_rel(
        l1: i32,
        l2: i32,
        no_nodes: i32,
        gauss_dist: f64,
        alpha1: &f64,
        alpha2: &f64,
    ) -> f64 {
        // Calculate the expansion coefficient for the overlap integral
        // between two contracted Gaussian functions.
        //
        // # Arguments
        // ----------
        // l1 : i32
        //    Angular momentum of the first Gaussian function.
        // l2 : i32
        //   Angular momentum of the second Gaussian function.
        // no_nodes : i32
        //   Number of nodes in Hermite (depends on type of int, e.g. always zero for overlap).
        // gauss_dist : f64
        //   Distance between the two Gaussian functions (from the origin)
        // alpha1 : f64
        //   Exponent of the first Gaussian function.
        // alpha2 : f64
        //   Exponent of the second Gaussian function.
        //

        let p_recip = (alpha1 + alpha2).recip();
        let q = alpha1 * alpha2 * p_recip;
        // Early return
        if no_nodes < 0 || no_nodes > (l1 + l2) {
            return 0.0;
        }

        match (no_nodes, l1, l2) {
            (0, 0, 0) => (-q * gauss_dist * gauss_dist).exp(),
            (_, _, 0) => {
                //* decrement index l1
                0.5 * p_recip
                    * Self::calc_recurr_rel(l1 - 1, l2, no_nodes - 1, gauss_dist, alpha1, alpha2)
                    - (q * gauss_dist / alpha1)
                        * Self::calc_recurr_rel(l1 - 1, l2, no_nodes, gauss_dist, alpha1, alpha2)
                    + (no_nodes + 1) as f64
                        * Self::calc_recurr_rel(
                            l1 - 1,
                            l2,
                            no_nodes + 1,
                            gauss_dist,
                            alpha1,
                            alpha2,
                        )
            }
            (_, _, _) => {
                //* decrement index l2
                0.5 * p_recip
                    * Self::calc_recurr_rel(l1, l2 - 1, no_nodes - 1, gauss_dist, alpha1, alpha2)
                    + (q * gauss_dist / alpha2)
                        * Self::calc_recurr_rel(l1, l2 - 1, no_nodes, gauss_dist, alpha1, alpha2)
                    + (no_nodes + 1) as f64
                        * Self::calc_recurr_rel(
                            l1,
                            l2 - 1,
                            no_nodes + 1,
                            gauss_dist,
                            alpha1,
                            alpha2,
                        )
            }
        }
    }
}

impl R_herm_aux_int {
    // Use Boys function to calculate the Hermite auxiliary integral
    // fn new() {
    //
    // }
}
