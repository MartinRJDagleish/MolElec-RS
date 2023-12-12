use crate::{basisset::{Shell, CGTO, PGTO}, molecule::cartesian_comp::CC_X};

///////////////////////////////////////
/// 1. Create function for invididual parts first
/// 2. Fuse into one big function to reduce redundant calcs
///////////////////////////////////////

pub fn calc_overlap_int_shell(shell1: Shell, shell2: Shell) -> f64 {
    let mut overlap_int = 0.0_f64;
    for cgto1 in shell1.cgto_iter() {
        for cgto2 in shell2.cgto_iter() {
            overlap_int += calc_overlap_int_cgto(cgto1, cgto2);
        }
    }
    overlap_int
}

fn calc_overlap_int_cgto(cgto1: &CGTO, cgto2: &CGTO) -> f64 {
    let mut overlap_int = 0.0_f64;
    for pgto1 in cgto1.pgto_iter() {
        for pgto2 in cgto2.pgto_iter() {
            overlap_int += calc_overlap_int_pgto(pgto1, pgto2);
        }
    }
    overlap_int
}

#[inline]
fn calc_overlap_int_pgto(pgto1: &PGTO, pgto2: &PGTO) -> f64 {
    let mut overlap_int = 0.0_f64;

    // let E_ab = EHermCoeff3D::new(pgto1, pgto2);
    // let S_ij = calc_E_herm_gauss_coeff(
    //         ang_mom_vec1[CC_X],
    //         ang_mom_vec2[CC_X],
    //         0,
    //         gauss1_center_pos[cart_coord] - gauss2_center_pos[cart_coord],
    //         alpha1,
    //         alpha2,
    //     );
    overlap_int
}

#[cfg(tests)]
mod tests {
    use super::*;

    // #[test]
    // fn
}
