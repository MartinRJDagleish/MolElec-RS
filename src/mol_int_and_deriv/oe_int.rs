use std::f64::consts::PI;

use crate::basisset::{CGTO, PGTO};
use crate::mol_int_and_deriv::recurrence_rel::EHermCoeff3D;
use crate::molecule::cartesian_comp::{CC_X, CC_Y, CC_Z};

use super::recurrence_rel::EHermCoeff1D;

lazy_static! {
    pub static ref PI_FAC_OVERL: f64 = PI.powf(1.5);
}

///////////////////////////////////////
/// 1. Create function for invididual parts first
/// 2. Fuse into one big function to reduce redundant calcs
///////////////////////////////////////

/// Calculate the overlap integral between two contracted Gaussian type orbitals (CGTOs)
/// Source: Helgaker -- Molecular Electronic Structure Theory
#[allow(non_snake_case)]
pub fn calc_overlap_int_cgto(cgto1: &CGTO, cgto2: &CGTO) -> f64 {
    let mut overlap_int = 0.0_f64;
    let vec_BA = cgto1.centre_pos().calc_vec_to_atom(cgto2.centre_pos());
    let ang_mom_vec1 = cgto1.ang_mom_vec();
    let ang_mom_vec2 = cgto2.ang_mom_vec();
    for pgto1 in cgto1.pgto_iter() {
        for pgto2 in cgto2.pgto_iter() {
            let E_ab = EHermCoeff3D::new(pgto1.alpha(), pgto2.alpha(), &vec_BA);
            let E_to_S_fac = *PI_FAC_OVERL * (1.0 / (pgto1.alpha() + pgto2.alpha()).powf(1.5));
            overlap_int += pgto1.norm_const()
                * pgto2.norm_const()
                * pgto1.pgto_coeff()
                * pgto2.pgto_coeff()
                * E_to_S_fac
                * E_ab.calc_recurr_rel(ang_mom_vec1, ang_mom_vec2, 0, 0);
        }
    }
    overlap_int // matrix element S_μν
}

#[allow(non_snake_case)]
#[inline(always)]
fn calc_T_cart_comp_pgto(pgto1: &PGTO, pgto2: &PGTO, l1: i32, l2: i32, vec_BA_comp: f64) -> f64 {
    let T_fac1 = -2.0 * pgto1.alpha() * pgto1.alpha();
    let T_fac2 = pgto2.alpha() * (2.0 * l2 as f64 + 1.0);
    let T_fac3 = -0.5 * l2 as f64 * (l2 + 1) as f64;
    let oo_alph_p = 1.0 / (pgto1.alpha() + pgto2.alpha());

    let T_ij = EHermCoeff1D::new(pgto1.alpha(), pgto2.alpha(), oo_alph_p, vec_BA_comp);
    let (E_ij_pl_2, E_ij, E_ij_min_2) = T_ij.calc_recurr_rel_for_kin(l1, l2);

    // This does not include the PI factor to red. redundant calcs -> add in cgto func
    E_ij_pl_2 * T_fac1 + E_ij * T_fac2 + E_ij_min_2 * T_fac3
}

#[allow(non_snake_case)]
pub fn calc_kinetic_int_cgto(cgto1: &CGTO, cgto2: &CGTO) -> f64 {
    let mut kin_int = 0.0_f64;
    let vec_BA = cgto1.centre_pos().calc_vec_to_atom(cgto2.centre_pos());
    let ang_mom_vec1 = cgto1.ang_mom_vec();
    let ang_mom_vec2 = cgto2.ang_mom_vec();
    //TODO: Diagonal elements are correctly calc., but not the off-diagonal ones
    for pgto1 in cgto1.pgto_iter() {
        for pgto2 in cgto2.pgto_iter() {
            let E_ab = EHermCoeff3D::new(pgto1.alpha(), pgto2.alpha(), &vec_BA);
            let (E_ij, E_kl, E_mn) =
                E_ab.calc_recurr_rel_ret_indv_parts(ang_mom_vec1, ang_mom_vec2, 0, 0);
            let T_x = calc_T_cart_comp_pgto(
                pgto1,
                pgto2,
                ang_mom_vec1[CC_X],
                ang_mom_vec2[CC_X],
                vec_BA[CC_X],
            );
            let T_y = calc_T_cart_comp_pgto(
                pgto1,
                pgto2,
                ang_mom_vec1[CC_Y],
                ang_mom_vec2[CC_Y],
                vec_BA[CC_Y],
            );
            let T_z = calc_T_cart_comp_pgto(
                pgto1,
                pgto2,
                ang_mom_vec1[CC_Z],
                ang_mom_vec2[CC_Z],
                vec_BA[CC_Z],
            );
            kin_int += pgto1.norm_const()
                * pgto2.norm_const()
                * pgto1.pgto_coeff()
                * pgto2.pgto_coeff()
                * *PI_FAC_OVERL
                * (1.0 / (pgto1.alpha() + pgto2.alpha()).powf(1.5))
                * (T_x * E_kl * E_mn + E_ij * T_y * E_mn + E_ij * E_kl * T_z);
        }
    }
    kin_int
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::atom::Atom;

    #[test]
    fn test_calc_overlap_int_cgto() {
        // let test_at = Atom::new(0.0, 0.0, 0.0, 1, );
        // let pgto1 = PGTO::new(0.5, 1.0, &[0, 0, 0]);
        // let cgto1 = CGTO::new(
        //     vec![pgto1],
        //     [0, 0, 0],
        //     [0.0, 0.0, 0.0],
        // );
        // assert_eq!(overlap_int, 1.0);
    }
}
