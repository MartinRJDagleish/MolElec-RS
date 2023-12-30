use crate::basisset::BasisSet;
use crate::mol_int_and_deriv::{
    oe_int::{calc_kinetic_int_cgto, calc_overlap_int_cgto, calc_pot_int_cgto},
    te_int::calc_ERI_int_cgto,
};
use crate::molecule::Molecule;
use ndarray::Array2;

use super::CalcSettings;

/// ### Description
/// Calculate the 1e integrals for the given basis set and molecule.
/// Returns a tuple of the overlap, kinetic and potential energy matrices.
///
/// ### Note
/// This is not the non-redudant version of the integrals, i.e. each function for the computation gets called separately.
///
///
/// ### Arguments
/// * `basis` - The basis set.
/// * `mol` - The molecule.
///
///
pub fn calc_1e_int_matrs(
    basis: &BasisSet,
    mol: &Molecule,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let mut S_matr = Array2::<f64>::zeros((basis.no_bf(), basis.no_bf()));
    let mut T_matr = Array2::<f64>::zeros((basis.no_bf(), basis.no_bf()));
    let mut V_matr = Array2::<f64>::zeros((basis.no_bf(), basis.no_bf()));

    let mut mu_sh_offset: usize = 0;
    let mut nu_sh_offset: usize = 0;
    let no_shells = basis.no_shells();
    for sh_idx1 in 0..no_shells {
        let shell1 = basis.shell(sh_idx1);
        for sh_idx2 in 0..=sh_idx1 {
            let shell2 = basis.shell(sh_idx2);
            for (cgto_idx1, cgto1) in shell1.cgto_iter().enumerate() {
                let mu_idx = mu_sh_offset + cgto_idx1;
                for (cgto_idx2, cgto2) in shell2.cgto_iter().enumerate() {
                    let nu_idx = nu_sh_offset + cgto_idx2;
                    S_matr[(mu_idx, nu_idx)] = if mu_idx == nu_idx {
                        1.0
                    } else {
                        calc_overlap_int_cgto(cgto1, cgto2)
                    };
                    S_matr[(nu_idx, mu_idx)] = S_matr[(mu_idx, nu_idx)];
                    T_matr[(mu_idx, nu_idx)] = calc_kinetic_int_cgto(cgto1, cgto2);
                    T_matr[(nu_idx, mu_idx)] = T_matr[(mu_idx, nu_idx)];
                    V_matr[(mu_idx, nu_idx)] = calc_pot_int_cgto(cgto1, cgto2, mol);
                    V_matr[(nu_idx, mu_idx)] = V_matr[(mu_idx, nu_idx)];
                }
            }
            nu_sh_offset += basis.shell_len(sh_idx2);
        }
        nu_sh_offset = 0;
        mu_sh_offset += basis.shell_len(sh_idx1);
    }
    //

    /////////////////////////////////
    // 1st working verison, but not very elegant
    /////////////////////////////////
    // let mut mu_sh_offset: usize = 0;
    // let mut nu_sh_offset: usize = 0;
    // for shell1 in basis.shell_iter() {
    //     for shell2 in basis.shell_iter() {
    //         for (cgto_idx1, cgto1) in shell1.cgto_iter().enumerate() {
    //             let mu_idx = mu_sh_offset + cgto_idx1;
    //             for (cgto_idx2, cgto2) in shell2.cgto_iter().enumerate() {
    //                 let nu_idx = nu_sh_offset + cgto_idx2;
    //                 if mu_idx == nu_idx {
    //                     S_matr[(mu_idx, nu_idx)] = 1.0;
    //                     continue;
    //                 } else {
    //                     S_matr[(mu_idx, nu_idx)] = calc_overlap_int_cgto(cgto1, cgto2);
    //                 }
    //             }
    //         }
    //         nu_sh_offset += shell2.shell_len();
    //     }
    //     nu_sh_offset = 0;
    //     mu_sh_offset += shell1.shell_len();
    // }

    (S_matr, T_matr, V_matr)
}

#[inline(always)]
fn calc_cmp_idx(idx1: usize, idx2: usize) -> usize {
    if idx1 >= idx2 {
        idx1 * (idx1 + 1) / 2 + idx2
    } else {
        idx2 * (idx2 + 1) / 2 + idx1
    }
}

pub fn calc_2e_int_matr(basis: &BasisSet) -> Array2<f64> {
    // let ERI
    let (mut mu_sh_offset, mut nu_sh_offset, mut lambda_sh_offset, mut sigma_sh_offset) =
        (0, 0, 0, 0);
    let no_shells = basis.no_shells();

    for sh_idx1 in 0..no_shells {
        let shell1 = basis.shell(sh_idx1);
        for sh_idx2 in 0..=sh_idx1 {
            let shell2 = basis.shell(sh_idx2);
            for (cgto_idx1, cgto1) in shell1.cgto_iter().enumerate() {
                let mu_idx = mu_sh_offset + cgto_idx1;
                for (cgto_idx2, cgto2) in shell2.cgto_iter().enumerate() {
                    let nu_idx = nu_sh_offset + cgto_idx2;
                    if mu_idx >= nu_idx {
                        let mu_nu = calc_cmp_idx(mu_idx, nu_idx);
                        for sh_idx3 in 0..no_shells {
                            let shell3 = basis.shell(sh_idx1);
                            for sh_idx4 in 0..=sh_idx3 {
                                let shell4 = basis.shell(sh_idx4);
                                for (cgto_idx3, cgto3) in shell3.cgto_iter().enumerate() {
                                    let lambda_idx = lambda_sh_offset + cgto_idx3;
                                    for (cgto_idx4, cgto4) in shell4.cgto_iter().enumerate() {
                                        let sigma_idx = sigma_sh_offset + cgto_idx4;
                                        if lambda_idx >= sigma_idx {
                                            let lambda_sigma = calc_cmp_idx(lambda_idx, sigma_idx);
                                            if mu_nu >= lambda_sigma {
                                                // TODO: add logic for lambda and sigma offsets
                                                // create some form of container for ERI (potentially avoid an Array4 if possible)
                                                // Use Array1 with cmp_idx -> ERI struct with better indexing? 

                                                // G_matr[(mu_nu, lambda_sigma)] =
                                                //     calc_ERI_int_cgto(cgto1, cgto2, cgto3, cgto4);
                                                // G_matr[(lambda_sigma, mu_nu)] =
                                                //     G_matr[(mu_nu, lambda_sigma)];
                                            } else {
                                                continue;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        continue;
                    }
                }
            }
            nu_sh_offset += basis.shell_len(sh_idx2);
        }
        nu_sh_offset = 0;
        mu_sh_offset += basis.shell_len(sh_idx1);
    }

    todo!();
    // G_matr
}


/// This is the main RHF SCF function to be called and run the RHF SCF calculation
pub(crate) fn rhf_scf_normal(calculation_settings: CalcSettings, max_scf_iter: usize) {

    for scf_iter in 0..max_scf_iter {
        println!("SCF Iteration: {}", scf_iter);
    }
    todo!();
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::Molecule;

    #[test]
    fn test_calc_1e_int_matrs() {
        println!("Test calc_1e_int_matrs");
        let mol = Molecule::new("data/xyz/water90.xyz", 0);
        let basis = BasisSet::new("STO-3G", &mol);

        let (S_matr, T_matr, V_matr) = calc_1e_int_matrs(&basis, &mol);
        println!("{}", S_matr);
        println!("{}", T_matr);
        println!("{}", V_matr);
    }
}
