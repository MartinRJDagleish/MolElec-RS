use std::f32::consts::E;
use std::f64::consts::PI;

use crate::mol_int_and_deriv::recurrence_rel::EHermCoeff3D;
use crate::{
    basisset::{Shell, CGTO, PGTO},
    molecule::cartesian_comp::{CC_X, CC_Y, CC_Z},
};

///////////////////////////////////////
/// 1. Create function for invididual parts first
/// 2. Fuse into one big function to reduce redundant calcs
///////////////////////////////////////

// pub fn calc_overlap_int_shell(shell1: Shell, shell2: Shell) -> f64 {
//     let mut overlap_int = 0.0_f64;
//     for cgto1 in shell1.cgto_iter() {
//         for cgto2 in shell2.cgto_iter() {
//             // overlap_int += calc_overlap_int_cgto(cgto1, cgto2);
//         }
//     }
//     overlap_int
// }

fn calc_overlap_int_cgto(cgto1: &CGTO, cgto2: &CGTO) -> f64 {
    lazy_static! {
        pub static ref PI_FAC: f64 = PI.powf(1.5);
    }
    let mut overlap_int = 0.0_f64;
    let vec_BA = cgto1.centre_pos().calc_vec_to_atom(cgto2.centre_pos());
    for pgto1 in cgto1.pgto_iter() {
        for pgto2 in cgto2.pgto_iter() {
            let E_ab = EHermCoeff3D::new(pgto1.alpha(), pgto2.alpha(), &vec_BA);
            let ang_mom_vec1 = cgto1.ang_mom_vec();
            let ang_mom_vec2 = cgto2.ang_mom_vec();
            let E_to_S_fac = *PI_FAC * (1.0 / (pgto1.alpha() + pgto2.alpha()).powf(1.5));
            overlap_int += E_to_S_fac * E_ab.calc_recurr_rel(ang_mom_vec1, ang_mom_vec2, 0, 0);
        }
    }
    overlap_int
}

#[cfg(test)]
mod tests {
    use crate::molecule::atom::Atom;
    use super::*;

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
