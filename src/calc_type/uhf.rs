use super::{CalcSettings, EriArr1, SCF};
use crate::{
    basisset::BasisSet,
    calc_type::{
        rhf::{calc_1e_int_matrs, calc_2e_int_matr, inv_ssqrt},
        DIIS,
    },
    mol_int_and_deriv::te_int::calc_schwarz_est_int,
    molecule::Molecule,
    print_utils::{fmt_f64, print_rhf::print_scf_header_and_settings},
};
use ndarray::parallel::prelude::*;
use ndarray::{linalg::general_mat_mul, s, Array1, Array2, Zip};
use ndarray_linalg::{Eigh, UPLO};

#[allow(unused)]
fn uhf_scf_normal(
    calc_sett: &CalcSettings,
    exec_times: &mut crate::print_utils::ExecTimes,
    mol: &Molecule,
    basis: &BasisSet,
) -> SCF {
    print_scf_header_and_settings(calc_sett, crate::calc_type::CalcType::UHF);
    let mut is_scf_conv = false;
    let mut scf = SCF::default();

    let no_elec_half = mol.no_elec() / 2;
    let (no_alpha, no_beta) = if mol.no_elec() % 2 == 0 {
        (no_elec_half, no_elec_half)
    } else {
        (no_elec_half + 1, no_elec_half)
    };

    let mut diis_alph;
    let mut diis_beta;
    if calc_sett.use_diis {
        diis_alph = Some(DIIS::new(&calc_sett.diis_sett, [no_alpha, no_alpha]));
        diis_beta = Some(DIIS::new(&calc_sett.diis_sett, [no_beta, no_beta]));
    } else {
        diis_alph = None;
        diis_beta = None;
    }

    let V_nuc: f64 = if mol.no_atoms() > 100 {
        mol.calc_core_potential_par()
    } else {
        mol.calc_core_potential_ser()
    };

    println!("Calculating 1e integrals ...");
    let (S_matr, H_core) = calc_1e_int_matrs(basis, mol);
    println!("FINSIHED calculating 1e integrals ...");

    let eri_opt;
    let schwarz_est_matr;
    if calc_sett.use_direct_scf {
        eri_opt = None;

        println!("Calculating Schwarz int estimates ...");
        schwarz_est_matr = Some(calc_schwarz_est_int(basis));
        println!("FINISHED Schwarz int estimates ...");
    } else {
        schwarz_est_matr = None;

        println!("Calculating 2e integrals ...");
        eri_opt = Some(calc_2e_int_matr(basis));
        println!("FINSIHED calculating 2e integrals ...");
    }

    let S_matr_inv_sqrt = inv_ssqrt(&S_matr, UPLO::Upper);

    // Init matrices for SCF loop
    let (mut orb_ener_alph, mut C_matr_MO_alph) = init_diag_F_matr(&H_core, &S_matr_inv_sqrt);
    let mut C_matr_AO_alph = S_matr_inv_sqrt.dot(&C_matr_MO_alph);

    let mut C_matr_MO_beta = C_matr_MO_alph.clone();
    let mut C_matr_AO_beta = C_matr_AO_alph.clone();
    let mut orb_ener_beta = orb_ener_alph.clone();

    let mut E_scf_prev = 0.0;

    let mut P_matr_alph = Array2::<f64>::zeros((basis.no_bf(), basis.no_bf()));
    build_P_matr_uhf(&mut P_matr_alph, &C_matr_AO_alph, no_alpha);
    let mut P_matr_beta = Array2::<f64>::zeros((basis.no_bf(), basis.no_bf()));
    build_P_matr_uhf(&mut P_matr_beta, &C_matr_AO_beta, no_beta);
    let mut P_matr_old_alph = P_matr_alph.clone();
    let mut P_matr_old_beta = P_matr_beta.clone();

    let mut diis_str = "";

    // Initial guess -> H_core
    let mut F_matr_alph = Array2::<f64>::zeros((basis.no_bf(), basis.no_bf()));
    let mut F_matr_beta = Array2::<f64>::zeros((basis.no_bf(), basis.no_bf()));

    let mut F_matr_pr_alph;
    let mut F_matr_pr_beta;

    println!(
        "{:>3} {:^20} {:^20} {:^20} {:^20}",
        "Iter", "E_scf", "E_tot", "ΔE", "RMS(|FPS - SPF|)"
    );

    for scf_iter in 1..=calc_sett.max_scf_iter {
        /// direct or indirect scf
        match eri_opt {
            Some(ref eri) => {
                calc_new_F_matr_ind_scf_uhf(
                    &mut F_matr_alph,
                    &H_core,
                    &P_matr_alph,
                    &P_matr_beta,
                    eri,
                    true,
                );
                calc_new_F_matr_ind_scf_uhf(
                    &mut F_matr_beta,
                    &H_core,
                    &P_matr_alph,
                    &P_matr_beta,
                    eri,
                    false,
                );
            }
            None => {
                todo!()
            }
        }
        let E_scf_curr = calc_E_scf_uhf(
            &P_matr_alph,
            &P_matr_beta,
            &H_core,
            &F_matr_alph,
            &F_matr_beta,
        );
        scf.E_tot_conv = E_scf_curr + V_nuc;
        let fps_comm_alph = DIIS::calc_FPS_comm(&F_matr_alph, &P_matr_alph, &S_matr);
        let fps_comm_beta = DIIS::calc_FPS_comm(&F_matr_beta, &P_matr_beta, &S_matr);

        F_matr_pr_alph = S_matr_inv_sqrt.dot(&F_matr_alph).dot(&S_matr_inv_sqrt);
        F_matr_pr_beta = S_matr_inv_sqrt.dot(&F_matr_beta).dot(&S_matr_inv_sqrt);

        if calc_sett.use_diis {
            let repl_idx = (scf_iter - 1) % calc_sett.diis_sett.diis_max; // always start with 0
            let err_matr_alph = S_matr_inv_sqrt.dot(&fps_comm_alph).dot(&S_matr_inv_sqrt);
            let err_matr_beta = S_matr_inv_sqrt.dot(&fps_comm_beta).dot(&S_matr_inv_sqrt);
            diis_alph
                .as_mut()
                .unwrap()
                .push_to_ring_buf(&F_matr_alph, &err_matr_alph, repl_idx);
            diis_beta
                .as_mut()
                .unwrap()
                .push_to_ring_buf(&F_matr_beta, &err_matr_beta, repl_idx);

            if scf_iter >= calc_sett.diis_sett.diis_min {
                let err_set_len = std::cmp::min(calc_sett.diis_sett.diis_max, scf_iter);
                F_matr_pr_alph = diis_alph.as_ref().unwrap().run_DIIS(err_set_len);
                F_matr_pr_beta = diis_beta.as_ref().unwrap().run_DIIS(err_set_len);
                diis_str = "DIIS";
            }
        }

        (orb_ener_alph, C_matr_MO_alph) = F_matr_pr_alph.eigh(UPLO::Upper).unwrap();
        C_matr_AO_alph = S_matr_inv_sqrt.dot(&C_matr_MO_alph);
        (orb_ener_beta, C_matr_MO_beta) = F_matr_pr_beta.eigh(UPLO::Upper).unwrap();
        C_matr_AO_beta = S_matr_inv_sqrt.dot(&C_matr_MO_beta);

        let delta_E = E_scf_curr - E_scf_prev;
        let rms_comm_val = ((fps_comm_alph.par_iter().map(|x| x * x).sum::<f64>()
            / fps_comm_alph.len() as f64)
            .sqrt()
            + (fps_comm_beta.par_iter().map(|x| x * x).sum::<f64>() / fps_comm_beta.len() as f64)
                .sqrt())
            * 0.5;

        // let rms_comm_val = 0.0;

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

        if (delta_E.abs() < calc_sett.e_diff_thrsh) && (rms_comm_val < calc_sett.commu_conv_thrsh) {
            // scf.tot_scf_iter = scf_iter;
            // scf.E_scf_conv = E_scf_curr;
            // scf.C_matr_conv = C_matr_AO.clone();
            // scf.P_matr_conv = P_matr.clone();
            // scf.orb_energies_conv = orb_ener.clone();
            println!("\nSCF CONVERGED!\n");
            is_scf_conv = true;
            break;
        } else if scf_iter == calc_sett.max_scf_iter {
            println!("\nSCF DID NOT CONVERGE!\n");
            break;
        }

        E_scf_prev = E_scf_curr;
        P_matr_old_alph = P_matr_alph.clone();
        P_matr_old_beta = P_matr_beta.clone();
        build_P_matr_uhf(&mut P_matr_alph, &C_matr_AO_alph, no_alpha);
        build_P_matr_uhf(&mut P_matr_beta, &C_matr_MO_beta, no_beta);
        if calc_sett.use_direct_scf {
            todo!()
            // delta_P_matr = Some((&P_matr - &P_matr_old).to_owned());
        }
    }

    scf
}

fn init_diag_F_matr(
    F_matr: &Array2<f64>,
    S_matr_inv_sqrt: &Array2<f64>,
) -> (Array1<f64>, Array2<f64>) {
    let F_matr_pr = S_matr_inv_sqrt.dot(F_matr).dot(S_matr_inv_sqrt);
    let (orb_ener, C_matr_MO) = F_matr_pr.eigh(UPLO::Upper).unwrap();
    (orb_ener, C_matr_MO)
}

fn build_P_matr_uhf(P_matr_spin: &mut Array2<f64>, C_matr_MO_spin: &Array2<f64>, n_orb: usize) {
    let C_occ = C_matr_MO_spin.slice(s![.., ..n_orb]);
    general_mat_mul(1.0_f64, &C_occ, &C_occ.t(), 0.0_f64, P_matr_spin);
}

fn divmod(n: usize, d: usize) -> (usize, usize) {
    (n / d, n % d)
}

fn calc_E_scf_uhf(
    P_matr_alph: &Array2<f64>,
    P_matr_beta: &Array2<f64>,
    H_core: &Array2<f64>,
    F_matr_alph: &Array2<f64>,
    F_matr_beta: &Array2<f64>,
) -> f64 {
    Zip::from(P_matr_alph)
        .and(P_matr_beta)
        .and(H_core)
        .and(F_matr_alph)
        .and(F_matr_beta)
        .into_par_iter()
        .map(|(&P_alph, &P_beta, &H, &F_alph, &F_beta)| {
            P_alph * (H + F_alph) + P_beta * (H + F_beta)
        })
        .sum::<f64>()
        * 0.5
}

fn calc_new_F_matr_ind_scf_uhf(
    F_matr_spin: &mut Array2<f64>,
    H_core: &Array2<f64>,
    P_matr_alph: &Array2<f64>,
    P_matr_beta: &Array2<f64>,
    eri: &EriArr1,
    is_alpha: bool,
) {
    F_matr_spin.assign(H_core);
    let no_bf = F_matr_spin.nrows();

    if is_alpha {
        // Fock matrix for alpha spin
        for mu in 0..no_bf {
            for nu in 0..=mu {
                for lambda in 0..no_bf {
                    for sigma in 0..no_bf {
                        let coul_P_mat_val =
                            P_matr_alph[(lambda, sigma)] + P_matr_beta[(lambda, sigma)];
                        F_matr_spin[(mu, nu)] += coul_P_mat_val * eri[(mu, nu, lambda, sigma)]
                            - eri[(mu, sigma, lambda, nu)] * P_matr_alph[(lambda, sigma)];
                    }
                }
                F_matr_spin[(nu, mu)] = F_matr_spin[(mu, nu)];
            }
        }
    } else {
        // Fock matrix for beta spin
        for mu in 0..no_bf {
            for nu in 0..=mu {
                for lambda in 0..no_bf {
                    for sigma in 0..no_bf {
                        let coul_P_mat_val =
                            P_matr_alph[(lambda, sigma)] + P_matr_beta[(lambda, sigma)];
                        F_matr_spin[(mu, nu)] += coul_P_mat_val * eri[(mu, nu, lambda, sigma)]
                            - eri[(mu, sigma, lambda, nu)] * P_matr_beta[(lambda, sigma)];
                    }
                }
                F_matr_spin[(nu, mu)] = F_matr_spin[(mu, nu)];
            }
        }
    }
}

fn calc_new_F_matr_dir_scf_uhf(
    F_matr: &mut Array2<f64>,
    delta_P_matr: &Array2<f64>,
    Schwarz_est_int: &Array2<f64>,
    basis: &BasisSet,
) {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{calc_type::DiisSettings, print_utils::ExecTimes};

    #[test]
    fn test_uhf_no_diis_indir_scf() {
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
        uhf_scf_normal(&calc_sett, &mut exec_times, &mol, &basis);
    }

    // #[test]
    // fn test_uhf_diis_indir_scf() {
    //     let mol = Molecule::new("data/xyz/water90.xyz", 0);
    //     let basis = BasisSet::new("STO-3G", &mol);
    //     let calc_sett = CalcSettings {
    //         max_scf_iter: 100,
    //         e_diff_thrsh: 1e-8,
    //         commu_conv_thrsh: 1e-8,
    //         use_diis: true,
    //         use_direct_scf: false,
    //         diis_sett: DiisSettings {
    //             diis_min: 2,
    //             diis_max: 6,
    //         },
    //     };
    //     let mut exec_times = ExecTimes::new();
    //     uhf_scf_normal(&calc_sett, &mut exec_times, &mol, &basis);
    // }
}
