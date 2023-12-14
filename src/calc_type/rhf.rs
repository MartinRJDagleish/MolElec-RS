// use crate::molecule::Molecule;
// use crate::calc_type::SCF;

// pub fn calc_RHF_w_wo_DIIS(&mut scf: SCF, &mol: Molecule, &basisset: BasisSet) {
// }

use ndarray::Array2;

use crate::basisset::BasisSet;
use crate::mol_int_and_deriv::oe_int::calc_overlap_int_cgto;

// pub fn calc_1e_int_matrs(basis: &BasisSet) -> Array2<f64> {
//     let S_matr = Array2::<f64>::zeros((basis.no_bf(), basis.no_bf()));
//     let mut mu_idx: usize = 0;
//
//     for (sh_idx1, shell1) in basis.shell_iter().enumerate() {
//         for (sh_idx2, shell2) in basis.shell_iter().enumerate() {
//             for cgto1 in shell1.cgto_iter() {
//                 for cgto2 in shell2.cgto_iter() {
//                     // let mu_idx = 
//                     overlap_val = calc_overlap_int_cgto(cgto1, cgto2);
//                 }
//             }
//         }
//     }
//     
//     S_matr
// }


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calc_1e_int_matrs() {
        println!("Test calc_1e_int_matrs");
        // let test_matr = calc_1e_int_matrs();
        // println!("{:?}", test_matr);
    }
}
