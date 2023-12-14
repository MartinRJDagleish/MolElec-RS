use crate::basisset::BasisSet;
use crate::mol_int_and_deriv::oe_int::{calc_kinetic_int_cgto, calc_overlap_int_cgto};
use ndarray::Array2;

pub fn calc_1e_int_matrs(basis: &BasisSet) -> (Array2<f64>, Array2<f64>) {
    let mut S_matr = Array2::<f64>::zeros((basis.no_bf(), basis.no_bf()));
    let mut T_matr = Array2::<f64>::zeros((basis.no_bf(), basis.no_bf()));

    /////////////////////////////////
    // 2nd version
    /////////////////////////////////
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
                    T_matr[(mu_idx, nu_idx)] = calc_kinetic_int_cgto(cgto1, cgto2);
                    if mu_idx == nu_idx {
                        S_matr[(mu_idx, nu_idx)] = 1.0;
                        continue;
                    } else {
                        S_matr[(mu_idx, nu_idx)] = calc_overlap_int_cgto(cgto1, cgto2);
                        S_matr[(nu_idx, mu_idx)] = S_matr[(mu_idx, nu_idx)];
                        T_matr[(mu_idx, nu_idx)] = calc_kinetic_int_cgto(cgto1, cgto2);
                        T_matr[(nu_idx, mu_idx)] = T_matr[(mu_idx, nu_idx)];
                    }
                }
            }
            nu_sh_offset += shell2.shell_len();
        }
        nu_sh_offset = 0;
        mu_sh_offset += shell1.shell_len();
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

    (S_matr, T_matr)
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

        let (S_matr, T_matr)  = calc_1e_int_matrs(&basis);
        println!("{}", S_matr);
        println!("{}", T_matr);
    }
}
