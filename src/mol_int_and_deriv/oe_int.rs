use ndarray::{array, Array1};
use std::f64::consts::PI;
use std::vec;

use crate::basisset::{CGTO, PGTO};
use crate::mol_int_and_deriv::recurrence_rel::{EHermCoeff1D, EHermCoeff3D, RHermAuxInt};
use crate::molecule::{
    atom::Atom,
    cartesian_comp::{CC_X, CC_Y, CC_Z},
    Molecule,
};

lazy_static! {
    pub static ref PI_FAC_OVERL: f64 = PI * PI.sqrt();
    pub static ref TWO_PI: f64 = 2.0 * PI;
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
/// Calculate the kinetic energy integral between two contracted Gaussian type orbitals (CGTOs)
/// Source: Helgaker -- Molecular Electronic Structure Theory
/// Here is only only one Cartesian comp. for the kinetic energy integral
fn calc_T_cart_comp_pgto(pgto1: &PGTO, pgto2: &PGTO, l1: i32, l2: i32, vec_BA_comp: f64) -> f64 {
    let T_fac1 = -2.0 * pgto2.alpha() * pgto2.alpha();
    let T_fac2 = pgto2.alpha() * (2 * l2 + 1) as f64;
    let T_fac3 = -0.5 * l2 as f64 * (l2 + 1) as f64;
    let oo_alph_p = 1.0 / (pgto1.alpha() + pgto2.alpha());
    // let E_to_S_fac = (PI * oo_alph_p).sqrt();

    let T_ij = EHermCoeff1D::new(pgto1.alpha(), pgto2.alpha(), oo_alph_p, vec_BA_comp);
    let (E_ij_pl_2, E_ij, E_ij_min_2) = T_ij.calc_recurr_rel_for_kin(l1, l2);

    // NOTE: This does not include the PI factor to red. redundant calcs -> add in cgto func
    // Here would technically be a sqrt(π / p) missing
    // E_to_S_fac * (E_ij_pl_2 * T_fac1 + E_ij * T_fac2 + E_ij_min_2 * T_fac3)
    E_ij_pl_2 * T_fac1 + E_ij * T_fac2 + E_ij_min_2 * T_fac3
}

#[allow(non_snake_case)]
/// Calculate the kinetic energy integral between two contracted Gaussian type orbitals (CGTOs)
/// Source: Helgaker -- Molecular Electronic Structure Theory
/// Here is only only one Cartesian comp. for the kinetic energy integral
pub fn calc_kinetic_int_cgto(cgto1: &CGTO, cgto2: &CGTO) -> f64 {
    let mut kin_int = 0.0_f64;
    let vec_BA = cgto1.centre_pos().calc_vec_to_atom(cgto2.centre_pos());
    let ang_mom_vec1 = cgto1.ang_mom_vec();
    let ang_mom_vec2 = cgto2.ang_mom_vec();
    for pgto1 in cgto1.pgto_iter() {
        for pgto2 in cgto2.pgto_iter() {
            let E_ab = EHermCoeff3D::new(pgto1.alpha(), pgto2.alpha(), &vec_BA);
            let (E_ij, E_kl, E_mn) =
                E_ab.calc_recurr_rel_ret_indv_parts(cgto1.ang_mom_vec(), cgto2.ang_mom_vec(), 0, 0);
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
                * (T_x * E_kl * E_mn + E_ij * T_y * E_mn + E_ij * E_kl * T_z)
                * *PI_FAC_OVERL
                * (1.0 / (pgto1.alpha() + pgto2.alpha())).powf(1.5);
            // * PI
            // * (1.0 / (pgto1.alpha() + pgto2.alpha()))
        }
    }
    kin_int
}

/// 3D Potential energy integral between two CGTOs
///
/// V_Ne integral using the MMD integration scheme
///
/// This is not separable for the R_herm term, but is separable for the
/// E_herm
#[allow(non_snake_case)]
pub fn calc_pot_int_cgto(cgto1: &CGTO, cgto2: &CGTO, mol: &Molecule) -> f64 {
    let mut pot_int = 0.0_f64;
    let mut tmp;
    let vec_BA = cgto1.centre_pos().calc_vec_to_atom(cgto2.centre_pos());
    let ang_mom_vec1 = cgto1.ang_mom_vec();
    let ang_mom_vec2 = cgto2.ang_mom_vec();
    let t_max = ang_mom_vec1[CC_X] + ang_mom_vec2[CC_X];
    let u_max = ang_mom_vec1[CC_Y] + ang_mom_vec2[CC_Y];
    let v_max = ang_mom_vec1[CC_Z] + ang_mom_vec2[CC_Z];

    // TOOD: which loop order is better to use pgto/pgto/atom, atom/pgto/pgto ?
    for atom_C in mol.atoms_iter() {
        tmp = 0.0;
        for pgto1 in cgto1.pgto_iter() {
            for pgto2 in cgto2.pgto_iter() {
                let E_ab = EHermCoeff3D::new(pgto1.alpha(), pgto2.alpha(), &vec_BA);
                let vec_P = calc_vec_P(
                    pgto1.alpha(),
                    pgto2.alpha(),
                    cgto1.centre_pos(),
                    cgto2.centre_pos(),
                );
                let vec_CP = array![
                    vec_P[CC_X] - atom_C[CC_X],
                    vec_P[CC_Y] - atom_C[CC_Y],
                    vec_P[CC_Z] - atom_C[CC_Z]
                ];
                let tot_ang_mom = *cgto1.ang_mom_type() as i32 + *cgto2.ang_mom_type() as i32;

                let R_tuv = RHermAuxInt::new(tot_ang_mom, vec_CP, pgto1.alpha() + pgto2.alpha());
                let mut R_recurr_val = 0.0_f64;
                let mut E_ab_vals = [0.0_f64; 3];
                for t in 0..=t_max {
                    E_ab_vals[CC_X] =
                        E_ab.E_ij
                            .calc_recurr_rel(ang_mom_vec1[CC_X], ang_mom_vec2[CC_X], t, 0);
                    for u in 0..=u_max {
                        E_ab_vals[CC_Y] =
                            E_ab.E_kl
                                .calc_recurr_rel(ang_mom_vec1[CC_Y], ang_mom_vec2[CC_Y], u, 0);
                        for v in 0..=v_max {
                            E_ab_vals[CC_Z] = E_ab.E_mn.calc_recurr_rel(
                                ang_mom_vec1[CC_Z],
                                ang_mom_vec2[CC_Z],
                                v,
                                0,
                            );
                            R_recurr_val += E_ab_vals.iter().product::<f64>()
                                * R_tuv.calc_recurr_rel(t, u, v, 0);
                        }
                    }
                }
                tmp += pgto1.norm_const()
                    * pgto2.norm_const()
                    * pgto1.pgto_coeff()
                    * pgto2.pgto_coeff()
                    * R_recurr_val
                    * *TWO_PI
                    / (pgto1.alpha() + pgto2.alpha());
            }
        }
        pot_int -= atom_C.z_val() as f64 * tmp;
    }

    pot_int
}

/// Calculate the Gaussian product center using two atoms and the exponential factors α_1 and α_2
///
/// Formula: vec_P = (α_1 * vec_A + α_2 * vec_B ) / p
///
/// using p = α_1 + α_2
#[allow(non_snake_case)]
#[inline]
pub(crate) fn calc_vec_P(alpha1: f64, alpha2: f64, atom_A: &Atom, atom_B: &Atom) -> Array1<f64> {
    // one over p
    let oop = 1.0 / (alpha1 + alpha2);
    let vec_A = array![atom_A[CC_X], atom_A[CC_Y], atom_A[CC_Z]];
    let vec_B = array![atom_B[CC_X], atom_B[CC_Y], atom_B[CC_Z]];
    (alpha1 * vec_A + alpha2 * vec_B) * oop
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::basisset::parser::AngMomChar;
    use crate::molecule::PseElemSym;
    use approx::assert_relative_eq;

    fn init_two_diff_cgtos<'a>(atom1: &'a Atom, atom2: &'a Atom) -> (CGTO<'a>, CGTO<'a>) {
        let ang_mom_vec1: [i32; 3] = [0, 0, 0];
        let L_tot = ang_mom_vec1.iter().sum::<i32>();
        let pgto_vec1: Vec<PGTO> = vec![
            PGTO::new(130.7093214, 0.1543289673, &ang_mom_vec1, L_tot),
            PGTO::new(23.80886605, 0.5353281423, &ang_mom_vec1, L_tot),
            PGTO::new(6.443608313, 0.4446345422, &ang_mom_vec1, L_tot),
        ];
        let pgto_vec1_len = pgto_vec1.len();

        let ang_mom_vec2: [i32; 3] = [0, 0, 0];
        let pgto_vec2 = vec![
            PGTO::new(5.033151319, -0.09996722919, &ang_mom_vec2, L_tot),
            PGTO::new(1.169596125, 0.3995128261, &ang_mom_vec2, L_tot),
            PGTO::new(0.38038896, 0.7001154689, &ang_mom_vec2, L_tot),
        ];
        let pgto_vec2_len = pgto_vec2.len();

        let cgto1 = CGTO::new(pgto_vec1, pgto_vec1_len, AngMomChar::S, ang_mom_vec1, atom1);
        let cgto2 = CGTO::new(pgto_vec2, pgto_vec2_len, AngMomChar::S, ang_mom_vec1, atom2);
        (cgto1, cgto2)
    }

    fn init_two_same_cgtos(atom: &Atom) -> (CGTO, CGTO) {
        let ang_mom_vec1: [i32; 3] = [0, 0, 0];
        let l_tot = ang_mom_vec1.iter().sum::<i32>();
        let pgto_vec: Vec<PGTO> = vec![
            PGTO::new(130.7093214, 0.1543289673, &ang_mom_vec1, l_tot),
            PGTO::new(23.80886605, 0.5353281423, &ang_mom_vec1, l_tot),
            PGTO::new(6.443608313, 0.4446345422, &ang_mom_vec1, l_tot),
        ];
        let pgto_vec_len = pgto_vec.len();

        let cgto1 = CGTO::new(
            pgto_vec.clone(),
            pgto_vec_len,
            AngMomChar::S,
            ang_mom_vec1,
            atom,
        );
        let cgto2 = CGTO::new(
            pgto_vec.clone(),
            pgto_vec_len,
            AngMomChar::S,
            ang_mom_vec1,
            atom,
        );
        (cgto1, cgto2)
    }

    #[test]
    fn test_calc_overlap_int_cgto_test1() {
        let atom = Atom::new(0.0, 0.0, 0.0, 8, crate::molecule::PseElemSym::O);
        let (cgto1, cgto2) = init_two_same_cgtos(&atom);

        let overlap_val = calc_overlap_int_cgto(&cgto1, &cgto2);
        println!("Overlap: {}", overlap_val);
        const OVERLAP_VAL_REF_1: f64 = 1.0;
        assert_relative_eq!(overlap_val, OVERLAP_VAL_REF_1, epsilon = 1e-10);
    }

    #[test]
    fn test_calc_overlap_int_cgto_test2() {
        let atom = Atom::new(0.0, 0.0, 0.0, 8, crate::molecule::PseElemSym::O);
        let (cgto1, cgto2) = init_two_diff_cgtos(&atom, &atom);

        let overlap_val = calc_overlap_int_cgto(&cgto1, &cgto2);
        println!("Overlap: {}", overlap_val);
        const OVERLAP_VAL_REF_2: f64 =
            // 0.2367039365; // initial val
            0.2367039206; // for sto-3g-new
        assert_relative_eq!(overlap_val, OVERLAP_VAL_REF_2, epsilon = 1e-10);
    }

    #[test]
    fn test_calc_kinetic_int_cgto_1() {
        let atom = Atom::new(0.0, 0.0, 0.0, 8, PseElemSym::O);
        let (cgto1, cgto2) = init_two_same_cgtos(&atom);

        let kin_val = calc_kinetic_int_cgto(&cgto1, &cgto2);
        println!("Kinetic1: {}", kin_val);
        const KIN_VAL_REF: f64 =
            // 29.0031999455; // initial value with old STO-3G
            29.003204064678087;
        assert_relative_eq!(kin_val, KIN_VAL_REF, epsilon = 1e-8);
    }

    //
    // #[test]
    // fn test_calc_kinetic_int_cgto_2() {
    //     let atom = Atom::new(0.0, 0.0, 0.0, 8, PseElemSym::O);
    //     let (cgto1, cgto2) = init_two_diff_cgtos(&atom, &atom);
    //
    //     let kin_val = calc_kinetic_int_cgto(&cgto1, &cgto2);
    //     println!("Kinetic2: {}", kin_val);
    //     // const KIN_VAL_REF: f64 =
    //     //     // 29.0031999455; // initial value with old STO-3G
    //     //     29.003204064678087;
    //     // assert_relative_eq!(kin_val, KIN_VAL_REF, epsilon = 1e-10);
    // }

    #[test]
    fn test_calc_pot_int_cgto_test1() {
        let mol = Molecule::new("data/xyz/water90.xyz", 0);
        let atom = Atom::new(0.0, 0.0, 0.0, 8, PseElemSym::O);
        let (cgto1, cgto2) = init_two_same_cgtos(&atom);

        let pot_val = calc_pot_int_cgto(&cgto1, &cgto2, &mol);
        println!("Potential 1: {}", pot_val);
        // Reference value computed with TCF programme
        const POT_VAL_REF_1: f64 = -61.7209117080;
        assert_relative_eq!(pot_val, POT_VAL_REF_1, epsilon = 1e-8);
    }

    #[test]
    fn test_calc_pot_int_cgto_test2() {
        let mol = Molecule::new("data/xyz/water90.xyz", 0);
        let atom = Atom::new(0.0, 0.0, 0.0, 8, crate::molecule::PseElemSym::O);
        let (cgto1, cgto2) = init_two_diff_cgtos(&atom, &atom);

        let pot_val = calc_pot_int_cgto(&cgto1, &cgto2, &mol);
        println!("Potential 2: {}", pot_val);
        // Reference value computed with TCF programme
        const POT_VAL_REF_2: f64 = -7.4440339458;

        assert_relative_eq!(pot_val, POT_VAL_REF_2, epsilon = 1e-8);
    }
    
    // #[test]
    // fn test_calc_eri_cgto_test1() {
    //     let atom = Atom::new(0.0, 0.0, 0.0, 8, crate::molecule::PseElemSym::O);
    //     let (cgto1, cgto2) = init_two_same_cgtos(&atom);
    //     let (cgto3, cgto4) = init_two_same_cgtos(&atom);
    //
    //     let eri_val = calc_ERI_int_cgto(&cgto1, &cgto2, &cgto3, &cgto4);
    //     println!("ERI val: {}", eri_val);
    // }
}
