use crate::basisset::BasisSet;
use crate::calc_type::{EriArr1, DIIS};
use crate::mol_int_and_deriv::{
    oe_int::{calc_kinetic_int_cgto, calc_overlap_int_cgto, calc_pot_int_cgto},
    te_int::calc_ERI_int_cgto,
};
use crate::molecule::Molecule;
use crate::print_utils::{
    fmt_f64,
    print_rhf::print_scf_header_and_settings,
    ExecTimes,
};
use ndarray::parallel::prelude::*;
use ndarray::{s, Array, Array1, Array2, Zip};
use ndarray_linalg::{Eigh, UPLO};

use super::{CalcSettings, SCF};

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
pub fn calc_1e_int_matrs(basis: &BasisSet, mol: &Molecule) -> (Array2<f64>, Array2<f64>) {
    let mut S_matr = Array2::<f64>::zeros((basis.no_bf(), basis.no_bf()));
    let mut T_matr = Array2::<f64>::zeros((basis.no_bf(), basis.no_bf()));
    let mut V_matr = Array2::<f64>::zeros((basis.no_bf(), basis.no_bf()));

    for (sh_idx1, shell1) in basis.shell_iter().enumerate() {
        for sh_idx2 in 0..=sh_idx1 {
            let shell2 = basis.shell(sh_idx2);
            for (cgto_idx1, cgto1) in shell1.cgto_iter().enumerate() {
                let mu = basis.sh_len_offset(sh_idx1) + cgto_idx1;
                for (cgto_idx2, cgto2) in shell2.cgto_iter().enumerate() {
                    let nu = basis.sh_len_offset(sh_idx2) + cgto_idx2;

                    // Overlap
                    S_matr[(mu, nu)] = if mu == nu {
                        1.0
                    } else {
                        calc_overlap_int_cgto(cgto1, cgto2)
                    };
                    S_matr[(nu, mu)] = S_matr[(mu, nu)];

                    // Kinetic
                    T_matr[(mu, nu)] = calc_kinetic_int_cgto(cgto1, cgto2);
                    T_matr[(nu, mu)] = T_matr[(mu, nu)];

                    // Potential energy
                    V_matr[(mu, nu)] = calc_pot_int_cgto(cgto1, cgto2, mol);
                    V_matr[(nu, mu)] = V_matr[(mu, nu)];
                }
            }
        }
    }

    //////////////////////////////////////////////
    // 2nd version: working, but trying to use sh_len_offsets
    //////////////////////////////////////////////
    // let mut mu_sh_offset: usize = 0;
    // let mut nu_sh_offset: usize = 0;
    // let no_shells = basis.no_shells();
    // for sh_idx1 in 0..no_shells {
    //     let shell1 = basis.shell(sh_idx1);
    //     for sh_idx2 in 0..=sh_idx1 {
    //         let shell2 = basis.shell(sh_idx2);
    //         for (cgto_idx1, cgto1) in shell1.cgto_iter().enumerate() {
    //             let mu_idx = mu_sh_offset + cgto_idx1;
    //             for (cgto_idx2, cgto2) in shell2.cgto_iter().enumerate() {
    //                 let nu_idx = nu_sh_offset + cgto_idx2;
    //                 S_matr[(mu_idx, nu_idx)] = if mu_idx == nu_idx {
    //                     1.0
    //                 } else {
    //                     calc_overlap_int_cgto(cgto1, cgto2)
    //                 };
    //                 S_matr[(nu_idx, mu_idx)] = S_matr[(mu_idx, nu_idx)];
    //                 T_matr[(mu_idx, nu_idx)] = calc_kinetic_int_cgto(cgto1, cgto2);
    //                 T_matr[(nu_idx, mu_idx)] = T_matr[(mu_idx, nu_idx)];
    //                 V_matr[(mu_idx, nu_idx)] = calc_pot_int_cgto(cgto1, cgto2, mol);
    //                 V_matr[(nu_idx, mu_idx)] = V_matr[(mu_idx, nu_idx)];
    //             }
    //         }
    //         nu_sh_offset += basis.shell_len(sh_idx2);
    //     }
    //     nu_sh_offset = 0;
    //     mu_sh_offset += basis.shell_len(sh_idx1);
    // }
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

    // Return ovelap and core hamiltonian (T + V)
    (S_matr, T_matr + V_matr)
}

#[inline(always)]
pub fn calc_cmp_idx(idx1: usize, idx2: usize) -> usize {
    if idx1 >= idx2 {
        idx1 * (idx1 + 1) / 2 + idx2
    } else {
        idx2 * (idx2 + 1) / 2 + idx1
    }
}

pub fn calc_2e_int_matr(basis: &BasisSet) -> EriArr1 {
    let mut eri = EriArr1::new(basis.no_bf());
    let no_shells = basis.no_shells();

    for (sh_idx1, shell1) in basis.shell_iter().enumerate() {
        for sh_idx2 in 0..=sh_idx1 {
            let shell2 = basis.shell(sh_idx2);
            for (cgto_idx1, cgto1) in shell1.cgto_iter().enumerate() {
                let mu = basis.sh_len_offset(sh_idx1) + cgto_idx1;
                for (cgto_idx2, cgto2) in shell2.cgto_iter().enumerate() {
                    let nu = basis.sh_len_offset(sh_idx2) + cgto_idx2;

                    if mu >= nu {
                        let munu = calc_cmp_idx(mu, nu);

                        for sh_idx3 in 0..no_shells {
                            let shell3 = basis.shell(sh_idx3);

                            for sh_idx4 in 0..=sh_idx3 {
                                let shell4 = basis.shell(sh_idx4);

                                for (cgto_idx3, cgto3) in shell3.cgto_iter().enumerate() {
                                    let lambda = basis.sh_len_offset(sh_idx3) + cgto_idx3;

                                    for (cgto_idx4, cgto4) in shell4.cgto_iter().enumerate() {
                                        let sigma = basis.sh_len_offset(sh_idx4) + cgto_idx4;

                                        if lambda >= sigma {
                                            let lambsig = calc_cmp_idx(lambda, sigma);
                                            if munu >= lambsig {
                                                let cmp_idx = calc_cmp_idx(munu, lambsig);
                                                eri[cmp_idx] =
                                                    calc_ERI_int_cgto(cgto1, cgto2, cgto3, cgto4);
                                                // println!("{}: {}", cmp_idx, eri[cmp_idx]);
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
        }
    }

    eri
}

/// This is the main RHF SCF function to be called and run the RHF SCF calculation
/// ## Options:
/// - DIIS
/// - direct vs. indirect SCF
#[allow(unused)]
pub(crate) fn rhf_scf_normal(
    calc_sett: CalcSettings,
    exec_times: &mut ExecTimes,
    basis: &BasisSet,
    mol: &Molecule,
) -> SCF {
    // TODO:
    // - [ ] Print settings
    // - [ ] Print initial header
    print_scf_header_and_settings(&calc_sett);
    const show_all_conv_crit: bool = false;

    let mut is_scf_conv = false;
    let mut scf = SCF::default();
    let mut diis: Option<DIIS>;
    match calc_sett.use_diis {
        true => {
            diis = Some(DIIS::new(
                &calc_sett.diis_sett,
                [basis.no_bf(), basis.no_bf()],
            ));
        }
        false => {
            diis = None;
        }
    }

    let V_nuc: f64 = if mol.no_atoms() > 100 {
        mol.calc_core_potential_par()
    } else {
        mol.calc_core_potential_ser()
    };

    let (S_matr, H_core) = calc_1e_int_matrs(basis, mol);

    let mut eri: Option<EriArr1> = if calc_sett.use_direct_scf {
        None
    } else {
        Some(calc_2e_int_matr(basis))
    };

    let S_matr_inv_sqrt = inv_ssqrt(&S_matr, UPLO::Upper);

    // Init matrices for SCF loop
    let mut C_matr_AO;
    let mut C_matr_MO;
    let mut orb_ener;
    let mut E_scf_prev = 0.0;

    let mut P_matr = Array2::<f64>::zeros((basis.no_bf(), basis.no_bf()));
    let mut P_matr_old = P_matr.clone();
    let mut F_matr = H_core.clone();
    let mut F_matr_pr;
    let mut diis_str = "";


    // Print SCF iteration Header
    match show_all_conv_crit {
        true => {
            println!(
                "{:>3} {:^20} {:^20} {:^20} {:^20} {:^20}",
                "Iter", "E_scf", "E_tot", "RMS(P)", "ΔE", "RMS(|FPS - SPF|)"
            );
        }
        false => {
            println!(
                "{:>3} {:^20} {:^20} {:^20} {:^20}",
                "Iter", "E_scf", "E_tot", "ΔE", "RMS(|FPS - SPF|)"
            );
        }
    }
    for scf_iter in 0..=calc_sett.max_scf_iter {
        if scf_iter == 0 {
            F_matr_pr = S_matr_inv_sqrt.dot(&F_matr).dot(&S_matr_inv_sqrt);
            // println!("F_matr_pr:\n {}", &F_matr_pr);

            (orb_ener, C_matr_MO) = F_matr_pr.eigh(UPLO::Upper).unwrap();
            C_matr_AO = S_matr_inv_sqrt.dot(&C_matr_MO);

            calc_P_matr(&mut P_matr, &C_matr_AO, basis.no_occ());
        } else {
            calc_new_F_matr(&mut F_matr, &H_core, &P_matr, &eri);
            let E_scf_curr = calc_E_scf(&P_matr, &H_core, &F_matr);
            scf.E_tot_conv = E_scf_curr + V_nuc;
            let fps_comm = DIIS::calc_FPS_comm(&F_matr, &P_matr, &S_matr);

            F_matr_pr = S_matr_inv_sqrt.dot(&F_matr).dot(&S_matr_inv_sqrt);

            if calc_sett.use_diis {
                let repl_idx = (scf_iter - 1) % calc_sett.diis_sett.diis_max; // always start with 0
                let err_matr = S_matr_inv_sqrt.dot(&fps_comm).dot(&S_matr_inv_sqrt);
                diis.as_mut()
                    .unwrap()
                    .push_to_ring_buf(&F_matr_pr, &err_matr, repl_idx);

                if scf_iter >= calc_sett.diis_sett.diis_min {
                    // println!("{:^120}", "*** ↓ Using DIIS ↓ ***");
                    let err_set_len = std::cmp::min(calc_sett.diis_sett.diis_max, scf_iter);
                    F_matr_pr = diis.as_ref().unwrap().run_DIIS(err_set_len);
                    diis_str = "DIIS";
                }
            }

            (orb_ener, C_matr_MO) = F_matr_pr.eigh(UPLO::Upper).unwrap();
            C_matr_AO = S_matr_inv_sqrt.dot(&C_matr_MO);

            let delta_E = E_scf_curr - E_scf_prev;
            let rms_comm_val =
                (fps_comm.par_iter().map(|x| x * x).sum::<f64>() / fps_comm.len() as f64).sqrt();
            if show_all_conv_crit {
                let rms_p_val = calc_rms_2_matr(&P_matr, &P_matr_old.clone());
                println!(
                    "{:>3} {:>20.12} {:>20.12} {:>20.12} {:>20.12} {:>20.12}",
                    scf_iter, E_scf_curr, scf.E_tot_conv, rms_p_val, delta_E, rms_comm_val
                );
            } else {
                println!(
                    "{:>3} {:>20.12} {:>20.12} {} {} {:>10} ",
                    scf_iter,
                    E_scf_curr,
                    scf.E_tot_conv,
                    fmt_f64(delta_E, 20, 8, 2),
                    fmt_f64(rms_comm_val, 20, 8, 2),
                    diis_str
                );
                diis_str = "";
            }

            if (delta_E.abs() < calc_sett.e_diff_thrsh)
                && (rms_comm_val < calc_sett.commu_conv_thrsh)
            {
                scf.tot_scf_iter = scf_iter;
                scf.E_scf_conv = E_scf_curr;
                scf.C_matr_conv = C_matr_AO.clone();
                scf.P_matr_conv = P_matr.clone();
                scf.orb_energies_conv = orb_ener.clone();
                println!("\nSCF CONVERGED!\n");
                is_scf_conv = true;
                break;
            } else if scf_iter == calc_sett.max_scf_iter {
                println!("\nSCF DID NOT CONVERGE!\n");
                break;
            }
            E_scf_prev = E_scf_curr;
            P_matr_old = P_matr.clone();
            calc_P_matr(&mut P_matr, &C_matr_AO, basis.no_occ());
        }
    }

    if is_scf_conv {
        println!("{:*<55}", "");
        println!("* {:^51} *", "FINAL RESULTS");
        println!("{:*<55}", "");
        println!("  {:^50}", "SCF (in a.u.)");
        println!("  {:=^50}  ", "");
        println!("  {:<25}{:>25}", "Total SCF iterations:", scf.tot_scf_iter);
        println!("  {:<25}{:>25.18}", "Final SCF energy:", scf.E_scf_conv);
        println!("  {:<25}{:>25.18}", "Final tot. energy:", scf.E_tot_conv);
        println!("{:*<55}", "");
    }
    scf
}

fn calc_P_matr(P_matr: &mut Array2<f64>, C_matr: &Array2<f64>, no_occ: usize) {
    let C_occ = C_matr.slice(s![.., ..no_occ]);
    P_matr.assign(&(2.0 * C_occ.dot(&C_occ.t())));
}

fn calc_new_F_matr(
    F_matr: &mut Array2<f64>,
    H_core: &Array2<f64>,
    P_matr: &Array2<f64>,
    eri: &Option<EriArr1>,
) {
    F_matr.assign(H_core);
    let no_bf = F_matr.nrows();

    // If eri is passed then this implies indirect SCF,
    // else direct SCF is assumed
    match eri {
        Some(eri) => {
            for mu in 0..no_bf {
                for nu in 0..=mu {
                    for lambda in 0..no_bf {
                        for sigma in 0..no_bf {
                            F_matr[(mu, nu)] += P_matr[(lambda, sigma)]
                                * (eri[(mu, nu, lambda, sigma)]
                                    - 0.5 * eri[(mu, sigma, lambda, nu)]);
                        }
                    }
                    F_matr[(nu, mu)] = F_matr[(mu, nu)];
                }
            }
        }
        None => todo!(),
    }
}

fn calc_E_scf(P_matr: &Array2<f64>, H_core: &Array2<f64>, F_matr: &Array2<f64>) -> f64 {
    Zip::from(P_matr)
        .and(H_core)
        .and(F_matr)
        .into_par_iter()
        .map(|(p, h, f)| *p * (*h + *f))
        .sum::<f64>()
        * 0.5
}

fn calc_rms_2_matr(matr1: &Array2<f64>, matr2: &Array2<f64>) -> f64 {
    Zip::from(matr1)
        .and(matr2)
        .into_par_iter()
        .map(|(val1, val2)| (val1 - val2).powi(2))
        .sum::<f64>()
        / (matr1.len() as f64).sqrt()
}

fn inv_ssqrt(arr2: &Array2<f64>, uplo: UPLO) -> Array2<f64> {
    let (e, v) = arr2.eigh(uplo).unwrap();
    let e_inv_sqrt = Array1::from_iter(e.iter().map(|x| x.powf(-0.5)));
    let e_inv_sqrt_diag = Array::from_diag(&e_inv_sqrt);
    let result = v.dot(&e_inv_sqrt_diag).dot(&v.t());
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{calc_type::DiisSettings, molecule::Molecule};

    #[test]
    fn test_calc_1e_int_matrs() {
        println!("Test calc_1e_int_matrs");
        let mol = Molecule::new("data/xyz/water90.xyz", 0);
        let basis = BasisSet::new("STO-3G", &mol);

        let (S_matr, H_core) = calc_1e_int_matrs(&basis, &mol);
        println!("{}", S_matr);
        println!("{}", H_core);
    }

    #[test]
    fn test_calc_2e_int_matr() {
        println!("Test calc_2e_int_matr");
        let mol = Molecule::new("data/xyz/water90.xyz", 0);
        let basis = BasisSet::new("STO-3G", &mol);

        let eri = calc_2e_int_matr(&basis);
        println!("{:?}", eri);
        // for (idx, val) in eri.eri_arr.iter().enumerate() {
        //     println!("{}: {}", idx, val);
        // }
    }

    #[test]
    fn test_rhf_indir_no_diis() {
        let mol = Molecule::new("data/xyz/water90.xyz", 0);
        let basis = BasisSet::new("STO-3G", &mol);
        let calc_sett = CalcSettings {
            max_scf_iter: 100,
            e_diff_thrsh: 1e-8,
            commu_conv_thrsh: 1e-8,
            use_diis: false,
            use_direct_scf: false,
            diis_sett: DiisSettings {
                diis_min: 0,
                diis_max: 0,
            },
        };
        let mut exec_times = ExecTimes::new();

        let _scf = rhf_scf_normal(calc_sett, &mut exec_times, &basis, &mol);
        println!("{:?}", _scf);
    }

    #[test]
    fn test_rhf_indir_diis() {
        let mol = Molecule::new("data/xyz/water90.xyz", 0);
        let basis = BasisSet::new("STO-3G", &mol);
        let calc_sett = CalcSettings {
            max_scf_iter: 100,
            e_diff_thrsh: 1e-8,
            commu_conv_thrsh: 1e-6,
            use_diis: true,
            use_direct_scf: false,
            diis_sett: DiisSettings {
                diis_min: 2,
                diis_max: 6,
            },
        };
        let mut exec_times = ExecTimes::new();

        let _scf = rhf_scf_normal(calc_sett, &mut exec_times, &basis, &mol);
        println!("{:?}", _scf);
    }
}
