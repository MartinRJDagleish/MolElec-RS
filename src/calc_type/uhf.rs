use super::{CalcSettings, SCF};
use crate::{
    basisset::BasisSet,
    calc_type::{
        rhf::{calc_1e_int_matrs, calc_2e_int_matr, inv_ssqrt},
        DIIS,
    },
    mol_int_and_deriv::te_int::calc_schwarz_est_int,
    molecule::Molecule,
};
use ndarray::{linalg::general_mat_mul, s, Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};

fn uhf_scf_normal(
    calc_sett: &CalcSettings,
    exec_times: &mut crate::print_utils::ExecTimes,
    mol: &Molecule,
    basis: &BasisSet,
) -> SCF {
    let mut is_scf_conv = false;
    let mut scf = SCF::default();
    let mut diis: Option<DIIS> = if calc_sett.use_diis {
        Some(DIIS::new(
            &calc_sett.diis_sett,
            [basis.no_bf(), basis.no_bf()],
        ))
    } else {
        None
    };

    let no_elec_half = mol.no_elec() / 2;
    let (n_alpha, n_beta) = if mol.no_elec() % 2 == 0 {
        (no_elec_half, no_elec_half)
    } else {
        (no_elec_half + 1, no_elec_half)
    };

    let V_nuc: f64 = if mol.no_atoms() > 100 {
        mol.calc_core_potential_par()
    } else {
        mol.calc_core_potential_ser()
    };

    println!("Calculating 1e integrals ...");
    let (S_matr, H_core) = calc_1e_int_matrs(basis, mol);
    println!("FINSIHED calculating 1e integrals ...");

    let mut eri;
    let schwarz_est_matr;
    if calc_sett.use_direct_scf {
        eri = None;

        println!("Calculating Schwarz int estimates ...");
        schwarz_est_matr = Some(calc_schwarz_est_int(basis));
        println!("FINISHED Schwarz int estimates ...");
    } else {
        schwarz_est_matr = None;

        println!("Calculating 2e integrals ...");
        eri = Some(calc_2e_int_matr(basis));
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

    let mut P_matr_alph;
    build_P_matr_uhf(&mut P_matr_alph, &C_matr_AO_alph, n_alpha);
    let mut P_matr_beta;
    build_P_matr_uhf(&mut P_matr_beta, &C_matr_AO_beta, n_beta);
    let mut P_matr_old_alph = P_matr_alph.clone();
    let mut P_matr_old_beta = P_matr_beta.clone();

    let mut F_matr_pr;
    let mut diis_str = "";

    // Initial guess -> H_core
    let mut F_matr_alph;
    let mut F_matr_beta;

    // for scf_iter in 1..=calc_sett.max_scf_iter {
    //         /// direct or indirect scf
    //         match eri {
    //             Some(ref eri) => {
    //                 calc_new_F_matr_ind_scf(&mut F_matr, &H_core, &P_matr, eri);
    //             }
    //             None => {
    //                 calc_new_F_matr_dir_scf(
    //                     &mut F_matr,
    //                     &delta_P_matr,
    //                     schwarz_est_matr.as_ref().unwrap(),
    //                     basis,
    //                 );
    //             }
    //         }
    //         let E_scf_curr = calc_E_scf(&P_matr, &H_core, &F_matr);
    //         scf.E_tot_conv = E_scf_curr + V_nuc;
    //         let fps_comm = DIIS::calc_FPS_comm(&F_matr, &P_matr, &S_matr);
    //
    //         F_matr_pr = S_matr_inv_sqrt.dot(&F_matr).dot(&S_matr_inv_sqrt);
    //
    //         if calc_sett.use_diis {
    //             let repl_idx = (scf_iter - 1) % calc_sett.diis_sett.diis_max; // always start with 0
    //             let err_matr = S_matr_inv_sqrt.dot(&fps_comm).dot(&S_matr_inv_sqrt);
    //             diis.as_mut()
    //                 .unwrap()
    //                 .push_to_ring_buf(&F_matr_pr, &err_matr, repl_idx);
    //
    //             if scf_iter >= calc_sett.diis_sett.diis_min {
    //                 // println!("{:^120}", "*** ↓ Using DIIS ↓ ***");
    //                 let err_set_len = std::cmp::min(calc_sett.diis_sett.diis_max, scf_iter);
    //                 F_matr_pr = diis.as_ref().unwrap().run_DIIS(err_set_len);
    //                 diis_str = "DIIS";
    //             }
    //         }
    //
    //         (orb_ener, C_matr_MO) = F_matr_pr.eigh(UPLO::Upper).unwrap();
    //         C_matr_AO = S_matr_inv_sqrt.dot(&C_matr_MO);
    //
    //         let delta_E = E_scf_curr - E_scf_prev;
    //         let rms_comm_val =
    //             (fps_comm.par_iter().map(|x| x * x).sum::<f64>() / fps_comm.len() as f64).sqrt();
    //         if SHOW_ALL_CONV_CRIT {
    //             let rms_p_val = calc_rms_2_matr(&P_matr, &P_matr_old.clone());
    //             println!(
    //                 "{:>3} {:>20.12} {:>20.12} {:>20.12} {:>20.12} {:>20.12}",
    //                 scf_iter, E_scf_curr, scf.E_tot_conv, rms_p_val, delta_E, rms_comm_val
    //             );
    //         } else {
    //             println!(
    //                 "{:>3} {:>20.12} {:>20.12} {} {} {:>10} ",
    //                 scf_iter,
    //                 E_scf_curr,
    //                 scf.E_tot_conv,
    //                 fmt_f64(delta_E, 20, 8, 2),
    //                 fmt_f64(rms_comm_val, 20, 8, 2),
    //                 diis_str
    //             );
    //             diis_str = "";
    //         }
    //
    //         if (delta_E.abs() < calc_sett.e_diff_thrsh)
    //             && (rms_comm_val < calc_sett.commu_conv_thrsh)
    //         {
    //             scf.tot_scf_iter = scf_iter;
    //             scf.E_scf_conv = E_scf_curr;
    //             scf.C_matr_conv = C_matr_AO.clone();
    //             scf.P_matr_conv = P_matr.clone();
    //             scf.orb_energies_conv = orb_ener.clone();
    //             println!("\nSCF CONVERGED!\n");
    //             is_scf_conv = true;
    //             break;
    //         } else if scf_iter == calc_sett.max_scf_iter {
    //             println!("\nSCF DID NOT CONVERGE!\n");
    //             break;
    //         }
    //         E_scf_prev = E_scf_curr;
    //         P_matr_old = P_matr.clone();
    //         calc_P_matr_rhf(&mut P_matr, &C_matr_AO, basis.no_occ());
    //         delta_P_matr = (&P_matr - &P_matr_old).to_owned();
    //     
    // }

    scf
}

fn init_diag_F_matr(
    F_matr: &Array2<f64>,
    S_matr_inv_sqrt: &Array2<f64>,
) -> (Array1<f64>, Array2<f64>) {
    let F_matr_pr = S_matr_inv_sqrt.dot(F_matr).dot(S_matr_inv_sqrt);
    let (orb_ener, C_matr) = F_matr_pr.eigh(UPLO::Upper).unwrap();
    let C_matr = S_matr_inv_sqrt.dot(&C_matr);
    (orb_ener, C_matr)
}

fn build_P_matr_uhf(P_matr_spin: &mut Array2<f64>, C_matr_MO_spin: &Array2<f64>, n_orb: usize) {
    let C_occ = C_matr_MO_spin.slice(s![.., ..n_orb]);
    general_mat_mul(1.0_f64, &C_occ, &C_occ.t(), 0.0_f64, P_matr_spin);
}

fn divmod(n: usize, d: usize) -> (usize, usize) {
    (n / d, n % d)
}
