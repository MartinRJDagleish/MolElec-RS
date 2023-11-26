use crate::molecule::cartesian_comp::{CC_X, CC_Y, CC_Z};
use boys::micb25::boys;
use ndarray::{Array1, ArrayView1};
use ndarray_linalg::Norm;

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
    vec_BA_comp: f64, // x, y, or z component of the vector from B to A (A_i - B_i)
    mu: f64,          // alpha1 * alpha2 * one_over_alph_p
}

#[derive(Debug, Default)]
struct R_herm_aux_int {
    // Hermite auxiliary int for the Hermite expansion of Cartesian Gaussian functions
    // See: Molecular Electronic-Structure Theory, Helgaker, Jorgensen, Olsen, 2000,
    boys_values: Vec<f64>,
    vec_CP: Array1<f64>, // Vector from C to P (P - C)
    // dist_CP: f64,        // Distance between C and P
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
    fn new(alpha1: f64, alpha2: f64, vec_BA: ArrayView1<f64>) -> Self {
        let one_over_alph_p = 1.0 / (alpha1 + alpha2);
        let E_ij = E_herm_coeff_1d::new(alpha1, alpha2, one_over_alph_p, vec_BA[CC_X]);
        let E_kl = E_herm_coeff_1d::new(alpha1, alpha2, one_over_alph_p, vec_BA[CC_Y]);
        let E_mn = E_herm_coeff_1d::new(alpha1, alpha2, one_over_alph_p, vec_BA[CC_Z]);

        Self { E_ij, E_kl, E_mn }
    }

    fn calc_recurr_rel(&self, l1: i32, l2: i32, no_nodes: i32, deriv_deg: i32) -> f64 {
        let E_ij_val = self.E_ij.calc_recurr_rel(l1, l2, no_nodes, deriv_deg);
        let E_kl_val = self.E_kl.calc_recurr_rel(l1, l2, no_nodes, deriv_deg);
        let E_mn_val = self.E_mn.calc_recurr_rel(l1, l2, no_nodes, deriv_deg);
        E_ij_val * E_kl_val * E_mn_val // return product of all three components
    }
}

impl E_herm_coeff_1d {
    fn new(alpha1: f64, alpha2: f64, one_over_alph_p: f64, vec_BA_comp: f64) -> Self {
        let mu = alpha1 * alpha2 * one_over_alph_p;
        Self {
            alpha1,
            alpha2,
            one_over_alph_p,
            vec_BA_comp,
            mu,
        }
    }

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

        //-------------- WORKING IMPL FOR MOL INTS AND DERIVS -----------------
        match (l1, l2, no_nodes, deriv_deg) {
            // Bases cases; 0th order deriv and 1st order deriv
            (0, 0, 0, 0) => (-self.mu * self.vec_BA_comp * self.vec_BA_comp).exp(),
            (0, 0, 0, 1) => {
                // equiv to -2.0 * mu *  R_x * E_0^00 ↑
                -2.0 * self.mu
                    * self.vec_BA_comp
                    * (-self.mu * self.vec_BA_comp * self.vec_BA_comp).exp()
            }
            (0, 0, 0, _) => {
                -2.0 * self.mu
                    * (self.vec_BA_comp * self.calc_recurr_rel(0, 0, 0, deriv_deg - 1)
                        + deriv_deg as f64 * self.calc_recurr_rel(0, 0, 0, deriv_deg - 2))
            }
            (_, 0, _, _) => {
                0.5 * self.one_over_alph_p
                    * self.calc_recurr_rel(l1 - 1, l2, no_nodes - 1, deriv_deg)
                    - self.alpha2
                        * self.one_over_alph_p
                        * (self.vec_BA_comp * self.calc_recurr_rel(l1 - 1, l2, no_nodes, deriv_deg)
                            + deriv_deg as f64
                                * self.calc_recurr_rel(l1 - 1, l2, no_nodes, deriv_deg - 1))
                    + (no_nodes + 1) as f64
                        * self.calc_recurr_rel(l1 - 1, l2, no_nodes + 1, deriv_deg)
            }
            (_, _, _, _) => {
                0.5 * self.one_over_alph_p
                    * self.calc_recurr_rel(l1, l2 - 1, no_nodes - 1, deriv_deg)
                    + self.alpha1
                        * self.one_over_alph_p
                        * (self.vec_BA_comp * self.calc_recurr_rel(l1, l2 - 1, no_nodes, deriv_deg)
                            + deriv_deg as f64
                                * self.calc_recurr_rel(l1, l2 - 1, no_nodes, deriv_deg - 1))
                    + (no_nodes + 1) as f64
                        * self.calc_recurr_rel(l1, l2 - 1, no_nodes + 1, deriv_deg)
            }
        }
    }
}

impl R_herm_aux_int {
    // Use Boys function to calculate the Hermite auxiliary integral
    fn new(tot_ang_mom: i32, vec_CP: Array1<f64>, alph_p: f64) -> Self {
        let dist_CP = vec_CP.norm();
        let boys_inp = alph_p * dist_CP * dist_CP;
        // double the capacity to allow for higher ang. mom. values
        // let mut boys_values = Vec::with_capacity((2 * tot_ang_mom) as usize);

        // Version 1 (just call boys for all values)
        // for ang_mom in 0..=(2 * tot_ang_mom as usize) {
        //     boys_values.push(boys(ang_mom as u64, boys_inp));
        // }

        // Version 2 (use recursion relation)
        // -> Helgakaer: Molecular Electronic-Structure Theory, 2000
        // Upward recursion is unstable => use downward recursion
        let boys_capacity = (2 * tot_ang_mom) as usize + 1;
        let mut boys_values = vec![0.0; boys_capacity]; // zero initialized
        boys_values[boys_capacity - 1] = boys((boys_capacity as u64), boys_inp);
        for ang_mom in (1..boys_capacity).rev() {
            boys_values[ang_mom - 1] = (2.0 * boys_inp * boys_values[ang_mom] + (-boys_inp).exp())
                / ((2 * (ang_mom - 1)) as f64 + 1.0);
            // boys_values.push(new_boys_val);
        }

        Self {
            boys_values,
            vec_CP,
            alph_p,
        }
    }

    fn calc_recurr_rel(&self, t: i32, u: i32, v: i32, boys_order: i32) -> f64 {
        // Early return -> error in calc
        if t < 0 || v < 0 || u < 0 {
            return 0.0;
        }

        match (t, u, v) {
            (0, 0, 0) => {
                let min_fac = if boys_order % 2 == 0 { 1.0 } else { -1.0 };
                min_fac
                    * (2.0 * self.alph_p).powi(boys_order)
                    * self.boys_values[boys_order as usize]
            }
            // early return for t
            (1, _, _) => self.vec_CP[CC_X] * self.calc_recurr_rel(0, u, v, boys_order + 1),
            (t, _, _) if t > 1 => {
                (t - 1) as f64 * self.calc_recurr_rel(t - 2, u, v, boys_order + 1)
                    + self.vec_CP[CC_X] * self.calc_recurr_rel(t - 1, u, v, boys_order + 1)
            }

            // early return for u
            (_, 1, _) => self.vec_CP[CC_Y] * self.calc_recurr_rel(t, 0, v, boys_order + 1),
            (_, u, _) if u > 1 => {
                (u - 1) as f64 * self.calc_recurr_rel(t, u - 2, v, boys_order + 1)
                    + self.vec_CP[CC_Y] * self.calc_recurr_rel(t, u - 1, v, boys_order + 1)
            }

            // early return for v
            (_, _, 1) => self.vec_CP[CC_Z] * self.calc_recurr_rel(t, u, 0, boys_order + 1),
            (_, _, v) if v > 1 => {
                (v - 1) as f64 * self.calc_recurr_rel(t, u, v - 2, boys_order + 1)
                    + self.vec_CP[CC_Z] * self.calc_recurr_rel(t, u, v - 1, boys_order + 1)
            }
            _ => 0.0,
        }
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

    #[test]
    fn test_E_calc_recurr_rel_deriv_test1() {
        let l1 = 2;
        let l2 = 1;
        let no_nodes = 1;
        let deriv_deg = 2;
        let alpha1 = 15.5;
        let alpha2 = 10.3;
        let test_vec_AB = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        let mut E_ab = E_herm_coeff_3d::new(alpha1, alpha2, ArrayView1::from(&test_vec_AB));

        let result = E_ab.calc_recurr_rel(l1, l2, no_nodes, deriv_deg);
        println!("result: {}", result);
        assert_abs_diff_eq!(result, 0.0000021278111580, epsilon = 1e-10); // reference from TCF
    }

    #[test]
    fn test_E_calc_recurr_rel_deriv_test2() {
        let l1 = 2;
        let l2 = 1;
        let no_nodes = 2;
        let deriv_deg = 2;
        let alpha1 = 15.5;
        let alpha2 = 10.3;
        let test_vec_AB = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        let mut E_ab = E_herm_coeff_3d::new(alpha1, alpha2, ArrayView1::from(&test_vec_AB));

        let result = E_ab.calc_recurr_rel(l1, l2, no_nodes, deriv_deg);
        println!("result: {}", result);
        assert_abs_diff_eq!(result, 0.0000000000767396, epsilon = 1e-10); // reference from TCF
    }

    #[test]
    fn test_R_calc_recurr_rel_test1() {
        let alpha1 = 15.5;
        let alpha2 = 10.3;
        let t = 2;
        let u = 1;
        let v = 0;
        let boys_order = t + u + v;
        let p = alpha1 + alpha2;

        let test_vec_CP = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        let R_tuv = R_herm_aux_int::new(boys_order, test_vec_CP, p);

        let result = R_tuv.calc_recurr_rel(t, u, v, boys_order);
        println!("result: {}", result);
        assert_abs_diff_eq!(result, -218102.9044892094389070, epsilon = 1e-10); // reference from TCF
    }

    #[test]
    fn test_R_calc_recurr_rel_test2() {
        let alpha1 = 15.5;
        let alpha2 = 10.3;
        let t = 0;
        let u = 0;
        let v = 0;
        let boys_order = t + u + v;
        let p = alpha1 + alpha2;

        let test_vec_CP = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        let R_tuv = R_herm_aux_int::new(boys_order, test_vec_CP, p);

        let result = R_tuv.calc_recurr_rel(t, u, v, boys_order);
        println!("result: {}", result);
        assert_abs_diff_eq!(result, 0.4629516875224093, epsilon = 1e-10); // reference from TCF
    }

    #[test]
    fn test_R_calc_recurr_rel_test3() {
        let alpha1 = 15.5;
        let alpha2 = 10.3;
        let t = 2;
        let u = 2;
        let v = 2;
        let boys_order = t + u + v;
        let p = alpha1 + alpha2;

        let test_vec_CP = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        let R_tuv = R_herm_aux_int::new(boys_order, test_vec_CP, p);

        let result = R_tuv.calc_recurr_rel(t, u, v, boys_order);
        println!("result: {}", result);
        // assert_abs_diff_eq!(result, -218102.9044892094389070, epsilon = 1e-10); // reference from TCF
    }
}
