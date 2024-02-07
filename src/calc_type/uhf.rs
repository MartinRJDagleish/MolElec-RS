use super::{rhf::calc_cmp_idx, CalcSettings, EriArr1, HFMatrices, HF_Ref, SCF};
use crate::{
    basisset::BasisSet,
    calc_type::{
        rhf::{matr_inv_ssqrt, RHF},
        DIIS, HF,
    },
    mol_int_and_deriv::{
        oe_int::{calc_kinetic_int_cgto, calc_overlap_int_cgto, calc_pot_int_cgto},
        te_int::{calc_ERI_int_cgto, calc_schwarz_est_int, calc_schwarz_est_int_inp},
    },
    molecule::Molecule,
    print_utils::{fmt_f64, print_scf::print_scf_header_and_settings, ExecTimes},
};
use ndarray::parallel::prelude::*;
use ndarray::{linalg::general_mat_mul, s, Array1, Array2, Zip};
use ndarray_linalg::{Eigh, UPLO};

pub(crate) struct UHF {
    // Matrices needed for the SCF calculation
    hf_matrs: HFMatrices,

    // f64 values for SCF calculation
    E_scf_prev: f64,
    E_scf_curr: f64,
    E_tot_prev: f64,
    E_tot_curr: f64,
}

impl HF for UHF {
    fn run_scf(
        &mut self,
        calc_sett: &CalcSettings,
        exec_times: &mut ExecTimes,
        basis: &BasisSet,
        mol: &Molecule,
    ) -> SCF {
        print_scf_header_and_settings(calc_sett, crate::calc_type::HF_Ref::UHF_ref);

        let mut is_scf_conv = false;
        let mut scf = SCF::default();

        // TODO: [ ] account for Multiplicity != 1 -> differnt input handling
        let no_elec_half = mol.no_elec() / 2;
        let (no_alpha, no_beta) = if mol.no_elec() % 2 == 0 {
            (no_elec_half, no_elec_half)
        } else {
            (no_elec_half + 1, no_elec_half)
        };

        let mut diis_alph = if calc_sett.use_diis {
            Some(DIIS::new(
                &calc_sett.diis_sett,
                [basis.no_bf(), basis.no_bf()],
            ))
        } else {
            None
        };
        let mut diis_beta = diis_alph.clone();

        let V_nuc: f64 = if mol.no_atoms() > 100 {
            mol.calc_core_potential_par()
        } else {
            mol.calc_core_potential_ser()
        };

        // Calculate 1e ints
        exec_times.start("1e ints");
        self.calc_1e_int_matrs_inp(basis, mol);
        exec_times.stop("1e ints");

        // Calculate 2e ints / Schwarz estimates
        exec_times.start("2e ints / Schwarz esti.");
        self.dir_indir_scf_2e_matr(basis, calc_sett);
        exec_times.stop("2e ints / Schwarz esti.");

        println!(
            "{:>3} {:^20} {:^20} {:^20} {:^20}",
            "Iter", "E_scf", "E_tot", "ΔE", "RMS(|FPS - SPF|)"
        );

        let mut diis_str = "";
        for scf_iter in 0..=calc_sett.max_scf_iter {
            if scf_iter == 0 {
                self.hf_matrs.F_matr_alpha = self.hf_matrs.H_core_matr.clone();
                self.hf_matrs.F_matr_beta = Some(self.hf_matrs.H_core_matr.clone());

                self.hf_matrs.F_matr_pr_alpha = self
                    .hf_matrs
                    .S_matr_inv_sqrt
                    .dot(&self.hf_matrs.F_matr_alpha)
                    .dot(&self.hf_matrs.S_matr_inv_sqrt);
                self.hf_matrs.F_matr_pr_beta = Some(
                    self.hf_matrs
                        .S_matr_inv_sqrt
                        .dot(self.hf_matrs.F_matr_beta.as_ref().unwrap())
                        .dot(&self.hf_matrs.S_matr_inv_sqrt),
                );

                (self.hf_matrs.orb_ener_alpha, self.hf_matrs.C_matr_MO_alpha) =
                    self.hf_matrs.F_matr_pr_alpha.eigh(UPLO::Upper).unwrap();
                self.hf_matrs.C_matr_AO_alpha = self
                    .hf_matrs
                    .S_matr_inv_sqrt
                    .dot(&self.hf_matrs.C_matr_MO_alpha);

                let (orb_e_beta, C_mat_bet) = self
                    .hf_matrs
                    .F_matr_pr_beta
                    .as_ref()
                    .unwrap()
                    .eigh(UPLO::Upper)
                    .unwrap();
                self.hf_matrs.orb_ener_beta.replace(orb_e_beta);
                self.hf_matrs.C_matr_MO_beta.replace(C_mat_bet);

                self.hf_matrs.C_matr_AO_beta = Some(
                    self.hf_matrs
                        .S_matr_inv_sqrt
                        .dot(self.hf_matrs.C_matr_MO_beta.as_ref().unwrap()),
                );

                Self::calc_P_matr_uhf(
                    &mut self.hf_matrs.P_matr_alpha,
                    &self.hf_matrs.C_matr_AO_alpha,
                    no_alpha,
                );
                Self::calc_P_matr_uhf(
                    self.hf_matrs.P_matr_beta.as_mut().unwrap(),
                    self.hf_matrs.C_matr_AO_beta.as_ref().unwrap(),
                    no_beta,
                );
                if calc_sett.use_direct_scf {
                    self.hf_matrs.delta_P_matr_alpha = Some(self.hf_matrs.P_matr_alpha.clone());
                    self.hf_matrs.delta_P_matr_beta = self.hf_matrs.P_matr_beta.clone();
                }
            } else {
                /// Direct or Indirect SCF
                match self.hf_matrs.eri_opt {
                    // Indirect SCF
                    Some(ref eri) => {
                        calc_new_F_matr_ind_scf_uhf(
                            &mut self.hf_matrs.F_matr_alpha,
                            &self.hf_matrs.H_core_matr,
                            &self.hf_matrs.P_matr_alpha,
                            self.hf_matrs.P_matr_beta.as_ref().unwrap(),
                            eri,
                            true,
                        );
                        calc_new_F_matr_ind_scf_uhf(
                            self.hf_matrs.F_matr_beta.as_mut().unwrap(),
                            &self.hf_matrs.H_core_matr,
                            &self.hf_matrs.P_matr_alpha,
                            self.hf_matrs.P_matr_beta.as_ref().unwrap(),
                            eri,
                            false,
                        );
                    }
                    // Direct SCF
                    None => {
                        todo!("Direct SCF for UHF not yet implemented!")
                    }
                }
                self.E_scf_curr = calc_E_scf_uhf(
                    &self.hf_matrs.P_matr_alpha,
                    self.hf_matrs.P_matr_beta.as_ref().unwrap(),
                    &self.hf_matrs.H_core_matr,
                    &self.hf_matrs.F_matr_alpha,
                    &self.hf_matrs.F_matr_beta.as_ref().unwrap(),
                );
                self.E_tot_curr = self.E_scf_curr + V_nuc;
                // FPS - SPF 
                let fps_comm_alph = DIIS::calc_FPS_comm(
                    &self.hf_matrs.F_matr_alpha,
                    &self.hf_matrs.P_matr_alpha,
                    &self.hf_matrs.S_matr,
                );
                let fps_comm_beta = DIIS::calc_FPS_comm(
                    self.hf_matrs.F_matr_beta.as_ref().unwrap(),
                    self.hf_matrs.P_matr_beta.as_ref().unwrap(),
                    &self.hf_matrs.S_matr,
                );

                self.hf_matrs.F_matr_pr_alpha = self
                    .hf_matrs
                    .S_matr_inv_sqrt
                    .dot(&self.hf_matrs.F_matr_alpha)
                    .dot(&self.hf_matrs.S_matr_inv_sqrt);
                self.hf_matrs.F_matr_pr_beta = Some(
                    self.hf_matrs
                        .S_matr_inv_sqrt
                        .dot(self.hf_matrs.F_matr_beta.as_ref().unwrap())
                        .dot(&self.hf_matrs.S_matr_inv_sqrt),
                );

                if calc_sett.use_diis {
                    let replace_idx = (scf_iter - 1) % calc_sett.diis_sett.diis_max; // always start with 0
                    let err_matr_alph = self
                        .hf_matrs
                        .S_matr_inv_sqrt
                        .dot(&fps_comm_alph)
                        .dot(&self.hf_matrs.S_matr_inv_sqrt);
                    let err_matr_beta = self
                        .hf_matrs
                        .S_matr_inv_sqrt
                        .dot(&fps_comm_beta)
                        .dot(&self.hf_matrs.S_matr_inv_sqrt);
                    diis_alph.as_mut().unwrap().push_to_ring_buf(
                        &self.hf_matrs.F_matr_pr_alpha,
                        &err_matr_alph,
                        replace_idx,
                    );
                    diis_beta.as_mut().unwrap().push_to_ring_buf(
                        self.hf_matrs.F_matr_pr_beta.as_ref().unwrap(),
                        &err_matr_beta,
                        replace_idx,
                    );

                    if scf_iter >= calc_sett.diis_sett.diis_min {
                        let err_set_len = std::cmp::min(calc_sett.diis_sett.diis_max, scf_iter);
                        self.hf_matrs.F_matr_pr_alpha =
                            diis_alph.as_ref().unwrap().run_DIIS(err_set_len);
                        self.hf_matrs.F_matr_pr_beta =
                            Some(diis_beta.as_ref().unwrap().run_DIIS(err_set_len));
                        diis_str = "DIIS";
                    }
                }

                (self.hf_matrs.orb_ener_alpha, self.hf_matrs.C_matr_MO_alpha) =
                    self.hf_matrs.F_matr_pr_alpha.eigh(UPLO::Upper).unwrap();
                self.hf_matrs.C_matr_AO_alpha = self
                    .hf_matrs
                    .S_matr_inv_sqrt
                    .dot(&self.hf_matrs.C_matr_MO_alpha);

                let (orb_e_beta, C_mat_bet) = self
                    .hf_matrs
                    .F_matr_pr_beta
                    .as_mut()
                    .unwrap()
                    .eigh(UPLO::Upper)
                    .unwrap();
                self.hf_matrs.orb_ener_beta.replace(orb_e_beta);
                self.hf_matrs.C_matr_MO_beta.replace(C_mat_bet);

                self.hf_matrs.C_matr_AO_beta = Some(
                    self.hf_matrs
                        .S_matr_inv_sqrt
                        .dot(self.hf_matrs.C_matr_MO_beta.as_ref().unwrap()),
                );

                let delta_E = self.E_scf_curr - self.E_scf_prev;
                let rms_comm_val = calc_rms_comm_val_uhf(&fps_comm_alph, &fps_comm_beta);

                println!(
                    "{:>3} {:>20.12} {:>20.12} {} {} {:>10} ",
                    scf_iter,
                    self.E_scf_curr,
                    self.E_tot_curr,
                    fmt_f64(delta_E, 20, 8, 2),
                    fmt_f64(rms_comm_val, 20, 8, 2),
                    diis_str
                );
                diis_str = "";

                if scf_conv_check(calc_sett, delta_E, rms_comm_val) {
                    scf.tot_scf_iter = scf_iter;
                    scf.E_scf_conv = self.E_scf_curr;
                    scf.E_tot_conv = self.E_tot_curr;
                    scf.C_matr_conv_alpha = self.hf_matrs.C_matr_AO_alpha.clone();
                    scf.C_matr_conv_beta = self.hf_matrs.C_matr_AO_beta.clone();
                    scf.P_matr_conv_alpha = self.hf_matrs.P_matr_alpha.clone();
                    scf.P_matr_conv_beta = self.hf_matrs.P_matr_beta.clone();
                    scf.orb_E_conv_alph = self.hf_matrs.orb_ener_alpha.clone();
                    scf.orb_E_conv_beta = self.hf_matrs.orb_ener_beta.clone();
                    scf.hf_ref = HF_Ref::UHF_ref;
                    println!("\nSCF CONVERGED!\n");
                    is_scf_conv = true;
                    break;
                } else if scf_iter == calc_sett.max_scf_iter {
                    println!("\nSCF DID NOT CONVERGE!\n");
                    break;
                }

                self.E_scf_prev = self.E_scf_curr;
                self.hf_matrs.P_matr_prev_alpha = self.hf_matrs.P_matr_alpha.clone();
                self.hf_matrs.P_matr_prev_beta = self.hf_matrs.P_matr_beta.clone();
                Self::calc_P_matr_uhf(
                    &mut self.hf_matrs.P_matr_alpha,
                    &self.hf_matrs.C_matr_AO_alpha,
                    no_alpha,
                );
                Self::calc_P_matr_uhf(
                    self.hf_matrs.P_matr_beta.as_mut().unwrap(),
                    self.hf_matrs.C_matr_AO_beta.as_ref().unwrap(),
                    no_beta,
                );
                if calc_sett.use_direct_scf {
                    todo!()
                    // delta_P_matr = Some((&P_matr - &P_matr_old).to_owned());
                }
            }
        }

        if is_scf_conv {
            println!("{:*<55}", "");
            println!("* {:^51} *", "FINAL RESULTS");
            println!("{:*<55}", "");
            println!("  {:^50}", "UHF SCF (in a.u.)");
            println!("  {:=^50}  ", "");
            println!("  {:<25}{:>25}", "Total SCF iterations:", scf.tot_scf_iter);
            println!("  {:<25}{:>25.18}", "Final SCF energy:", scf.E_scf_conv);
            println!("  {:<25}{:>25.18}", "Final tot. energy:", scf.E_tot_conv);
            println!("{:*<55}", "");
        }
        scf
    }
}

impl UHF {
    pub fn new(basis: &BasisSet, calc_sett: &CalcSettings) -> Self {
        // UHF has always beta variables
        const CREATE_BETA_VARS: bool = true;

        Self {
            hf_matrs: HFMatrices::new(basis.no_bf(), calc_sett.use_direct_scf, CREATE_BETA_VARS),
            E_scf_prev: 0.0,
            E_scf_curr: 0.0,
            E_tot_prev: 0.0,
            E_tot_curr: 0.0,
        }
    }

    fn calc_1e_int_matrs_inp(&mut self, basis: &BasisSet, mol: &Molecule) {
        println!("Calculating 1e integrals ...");

        for (sh_idx1, shell1) in basis.shell_iter().enumerate() {
            for sh_idx2 in 0..=sh_idx1 {
                let shell2 = basis.shell(sh_idx2);
                for (cgto_idx1, cgto1) in shell1.cgto_iter().enumerate() {
                    let mu = basis.sh_len_offset(sh_idx1) + cgto_idx1;
                    for (cgto_idx2, cgto2) in shell2.cgto_iter().enumerate() {
                        let nu = basis.sh_len_offset(sh_idx2) + cgto_idx2;

                        // Overlap
                        self.hf_matrs.S_matr[(mu, nu)] = if mu == nu {
                            1.0
                        } else {
                            calc_overlap_int_cgto(cgto1, cgto2)
                        };
                        self.hf_matrs.S_matr[(nu, mu)] = self.hf_matrs.S_matr[(mu, nu)];

                        // Kinetic
                        self.hf_matrs.T_matr[(mu, nu)] = calc_kinetic_int_cgto(cgto1, cgto2);
                        self.hf_matrs.T_matr[(nu, mu)] = self.hf_matrs.T_matr[(mu, nu)];

                        // Potential energy
                        self.hf_matrs.V_ne_matr[(mu, nu)] = calc_pot_int_cgto(cgto1, cgto2, mol);
                        self.hf_matrs.V_ne_matr[(nu, mu)] = self.hf_matrs.V_ne_matr[(mu, nu)];
                    }
                }
            }
        }
        // println!("S_matr: \n{:>12.8}", &self.hf_matrs.S_matr);
        // println!("T_matr: \n{:>12.8}", &self.hf_matrs.T_matr);
        // println!("V_matr: \n{:>12.8}", &self.hf_matrs.V_ne_matr);

        Zip::from(self.hf_matrs.T_matr.view())
            .and(self.hf_matrs.V_ne_matr.view())
            .par_map_assign_into(&mut self.hf_matrs.H_core_matr, |&t, &v| t + v);
        println!("FINSIHED calculating 1e integrals ...");

        println!("Starting orthogonalization matrix calculation ...");
        self.hf_matrs.S_matr_inv_sqrt = matr_inv_ssqrt(&self.hf_matrs.S_matr, UPLO::Upper);
        println!("FINISHED orthogonalization matrix calculation ...");
    }

    fn calc_2e_int_matr_inp(eri_arr: &mut EriArr1, basis: &BasisSet) {
        let no_shells = basis.no_shells();

        for (sh_idx1, shell1) in basis.shell_iter().enumerate() {
            for (cgto_idx1, cgto1) in shell1.cgto_iter().enumerate() {
                let mu = basis.sh_len_offset(sh_idx1) + cgto_idx1;

                for sh_idx2 in 0..=sh_idx1 {
                    let shell2 = basis.shell(sh_idx2);
                    for (cgto_idx2, cgto2) in shell2.cgto_iter().enumerate() {
                        let nu = basis.sh_len_offset(sh_idx2) + cgto_idx2;

                        if mu >= nu {
                            let munu = calc_cmp_idx(mu, nu);

                            for sh_idx3 in 0..no_shells {
                                let shell3 = basis.shell(sh_idx3);
                                for (cgto_idx3, cgto3) in shell3.cgto_iter().enumerate() {
                                    let lambda = basis.sh_len_offset(sh_idx3) + cgto_idx3;

                                    for sh_idx4 in 0..=sh_idx3 {
                                        let shell4 = basis.shell(sh_idx4);
                                        for (cgto_idx4, cgto4) in shell4.cgto_iter().enumerate() {
                                            let sigma = basis.sh_len_offset(sh_idx4) + cgto_idx4;

                                            if lambda >= sigma {
                                                let lambsig = calc_cmp_idx(lambda, sigma);
                                                if munu >= lambsig {
                                                    let cmp_idx = calc_cmp_idx(munu, lambsig);
                                                    eri_arr[cmp_idx] = calc_ERI_int_cgto(
                                                        cgto1, cgto2, cgto3, cgto4,
                                                    );
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
    }

    fn dir_indir_scf_2e_matr(&mut self, basis: &BasisSet, calc_sett: &CalcSettings) {
        match calc_sett.use_direct_scf {
            true => {
                println!("Calculating Schwarz int estimates ...");
                calc_schwarz_est_int_inp(self.hf_matrs.schwarz_est.as_mut().unwrap(), basis);
                println!("FINISHED Schwarz int estimates ...");
            }
            false => {
                println!("Calculating 2e integrals ...");
                Self::calc_2e_int_matr_inp(self.hf_matrs.eri_opt.as_mut().unwrap(), basis);
                // println!("i j k l; ijkl: val");
                // for i in 0..basis.no_bf() {
                //     for j in 0..basis.no_bf() {
                //         for k in 0..basis.no_bf() {
                //             for l in 0..basis.no_bf() {
                //                 let ijkl = calc_cmp_idx(calc_cmp_idx(i, j), calc_cmp_idx(k, l));
                //                 println!("{:>3}{:>3}{:>3}{:>3}; {:>4}: {:>12.8}", i,j,k,l,ijkl, self.hf_matrs.eri_opt.as_ref().unwrap()[ijkl]);
                //             }
                //         }
                //     }
                // }
                println!("FINSIHED calculating 2e integrals ...");
            }
        }
    }

    fn calc_P_matr_uhf(P_matr_spin: &mut Array2<f64>, C_matr_AO_spin: &Array2<f64>, n_orb: usize) {
        let C_occ = C_matr_AO_spin.slice(s![.., ..n_orb]);
        general_mat_mul(1.0_f64, &C_occ, &C_occ.t(), 0.0_f64, P_matr_spin);
    }
    // pub fn run_scf(
    //     &mut self,
    //     calc_sett: &CalcSettings,
    //     exec_times: &mut crate::print_utils::ExecTimes,
    //     basis: &BasisSet,
    //     mol: &Molecule,
    // ) -> SCF {
    //     print_scf_header_and_settings(calc_sett, HF_Ref::UHF_ref);
    //
    //     let mut is_scf_conv = false;
    //     let mut scf = SCF::default();
    //     let mut diis: Option<DIIS> = if calc_sett.use_diis {
    //         Some(DIIS::new(
    //             &calc_sett.diis_sett,
    //             [basis.no_bf(), basis.no_bf()],
    //         ))
    //     } else {
    //         None
    //     };
    //
    //     let V_nuc: f64 = if mol.no_atoms() > 100 {
    //         mol.calc_core_potential_par()
    //     } else {
    //         mol.calc_core_potential_ser()
    //     };
    //
    //     // Calculate 1e ints
    //     exec_times.start("1e ints");
    //     self.calc_1e_int_matrs_inp(basis, mol);
    //     exec_times.stop("1e ints");
    //
    //     // Calculate 2e ints / Schwarz estimates
    //     exec_times.start("2e ints / Schwarz esti.");
    //     self.dir_indir_scf_2e_matr(basis, calc_sett);
    //     exec_times.stop("2e ints / Schwarz esti.");
    //
    //     // Initial guess -> H_core
    //     // TODO: [ ] replace with guess
    //     self.hf_matrs.F_matr_alpha = self.hf_matrs.H_core_matr.clone();
    //
    //     // Print SCF iteration Header
    //     println!(
    //         "{:>3} {:^20} {:^20} {:^20} {:^20}",
    //         "Iter", "E_scf", "E_tot", "ΔE", "RMS(|FPS - SPF|)"
    //     );
    //     let mut diis_str = "";
    //     for scf_iter in 0..=calc_sett.max_scf_iter {
    //         if scf_iter == 0 {
    //             self.hf_matrs.F_matr_pr_alpha = self
    //                 .hf_matrs
    //                 .S_matr_inv_sqrt
    //                 .dot(&self.hf_matrs.F_matr_alpha)
    //                 .dot(&self.hf_matrs.S_matr_inv_sqrt);
    //
    //             (self.hf_matrs.orb_ener_alpha, self.hf_matrs.C_matr_MO_alpha) =
    //                 self.hf_matrs.F_matr_pr_alpha.eigh(UPLO::Upper).unwrap();
    //             self.hf_matrs.C_matr_AO_alpha = self
    //                 .hf_matrs
    //                 .S_matr_inv_sqrt
    //                 .dot(&self.hf_matrs.C_matr_MO_alpha);
    //
    //             Self::calc_P_matr_rhf(
    //                 &mut self.hf_matrs.P_matr_alpha,
    //                 &self.hf_matrs.C_matr_AO_alpha,
    //                 basis.no_occ(),
    //             );
    //             if calc_sett.use_direct_scf {
    //                 self.hf_matrs.delta_P_matr_alpha = Some(self.hf_matrs.P_matr_alpha.clone());
    //             }
    //         } else {
    //             /// direct or indirect scf
    //             match self.hf_matrs.eri_opt {
    //                 Some(ref eri) => {
    //                     Self::calc_new_F_matr_ind_scf_rhf(
    //                         &mut self.hf_matrs.F_matr_alpha,
    //                         &self.hf_matrs.H_core_matr,
    //                         &self.hf_matrs.P_matr_alpha,
    //                         eri,
    //                     );
    //                 }
    //                 None => {
    //                     Self::calc_new_F_matr_dir_scf_rhf(
    //                         &mut self.hf_matrs.F_matr_alpha,
    //                         self.hf_matrs.delta_P_matr_alpha.as_ref().unwrap(),
    //                         self.hf_matrs.schwarz_est.as_ref().unwrap(),
    //                         basis,
    //                     );
    //                 }
    //             }
    //             self.E_scf_curr = Self::calc_E_scf_rhf(
    //                 &self.hf_matrs.P_matr_alpha,
    //                 &self.hf_matrs.H_core_matr,
    //                 &self.hf_matrs.F_matr_alpha,
    //             );
    //             self.E_tot_curr = self.E_scf_curr + V_nuc;
    //             // FPS - SPF
    //             let fps_comm = DIIS::calc_FPS_comm(
    //                 &self.hf_matrs.F_matr_alpha,
    //                 &self.hf_matrs.P_matr_alpha,
    //                 &self.hf_matrs.S_matr,
    //             );
    //
    //             // F' = S^(-1/2) * F * S^(-1/2)
    //             self.hf_matrs.F_matr_pr_alpha = self
    //                 .hf_matrs
    //                 .S_matr_inv_sqrt
    //                 .dot(&self.hf_matrs.F_matr_alpha)
    //                 .dot(&self.hf_matrs.S_matr_inv_sqrt);
    //
    //             if calc_sett.use_diis {
    //                 let repl_idx = (scf_iter - 1) % calc_sett.diis_sett.diis_max; // always start with 0
    //                 let err_matr = self
    //                     .hf_matrs
    //                     .S_matr_inv_sqrt
    //                     .dot(&fps_comm)
    //                     .dot(&self.hf_matrs.S_matr_inv_sqrt);
    //                 diis.as_mut().unwrap().push_to_ring_buf(
    //                     &self.hf_matrs.F_matr_pr_alpha,
    //                     &err_matr,
    //                     repl_idx,
    //                 );
    //
    //                 if scf_iter >= calc_sett.diis_sett.diis_min {
    //                     let err_set_len = std::cmp::min(calc_sett.diis_sett.diis_max, scf_iter);
    //                     self.hf_matrs.F_matr_pr_alpha =
    //                         diis.as_ref().unwrap().run_DIIS(err_set_len);
    //                     diis_str = "DIIS";
    //                 }
    //             }
    //
    //             (self.hf_matrs.orb_ener_alpha, self.hf_matrs.C_matr_MO_alpha) =
    //                 self.hf_matrs.F_matr_pr_alpha.eigh(UPLO::Upper).unwrap();
    //             self.hf_matrs.C_matr_AO_alpha = self
    //                 .hf_matrs
    //                 .S_matr_inv_sqrt
    //                 .dot(&self.hf_matrs.C_matr_MO_alpha);
    //
    //             let delta_E = self.E_scf_curr - self.E_scf_prev;
    //             let rms_comm_val = (fps_comm.par_iter().map(|x| x * x).sum::<f64>()
    //                 / fps_comm.len() as f64)
    //                 .sqrt();
    //             println!(
    //                 "{:>3} {:>20.12} {:>20.12} {} {} {:>10} ",
    //                 scf_iter,
    //                 self.E_scf_curr,
    //                 self.E_tot_curr,
    //                 fmt_f64(delta_E, 20, 8, 2),
    //                 fmt_f64(rms_comm_val, 20, 8, 2),
    //                 diis_str
    //             );
    //             diis_str = "";
    //
    //             if (delta_E.abs() < calc_sett.e_diff_thrsh)
    //                 && (rms_comm_val < calc_sett.commu_conv_thrsh)
    //             {
    //                 scf.tot_scf_iter = scf_iter;
    //                 scf.E_scf_conv = self.E_scf_curr;
    //                 scf.E_tot_conv = self.E_tot_curr;
    //                 scf.C_matr_conv_alph = self.hf_matrs.C_matr_AO_alpha.clone();
    //                 scf.P_matr_conv_alph = self.hf_matrs.P_matr_alpha.clone();
    //                 println!("P_matr conv:\n{:>12.8}", &self.hf_matrs.P_matr_alpha);
    //                 scf.C_matr_conv_beta = None;
    //                 scf.P_matr_conv_beta = None;
    //                 scf.orb_E_conv_alph = self.hf_matrs.orb_ener_alpha.clone();
    //                 println!("\nSCF CONVERGED!\n");
    //                 is_scf_conv = true;
    //                 break;
    //             } else if scf_iter == calc_sett.max_scf_iter {
    //                 println!("\nSCF DID NOT CONVERGE!\n");
    //                 break;
    //             }
    //             self.E_scf_prev = self.E_scf_curr;
    //             self.hf_matrs.P_matr_prev_alpha = self.hf_matrs.P_matr_alpha.clone();
    //             Self::calc_P_matr_rhf(
    //                 &mut self.hf_matrs.P_matr_alpha,
    //                 &self.hf_matrs.C_matr_AO_alpha,
    //                 basis.no_occ(),
    //             );
    //             if calc_sett.use_direct_scf {
    //                 Zip::from(&self.hf_matrs.P_matr_alpha.view())
    //                     .and(&self.hf_matrs.P_matr_prev_alpha.view())
    //                     .par_map_assign_into(
    //                         self.hf_matrs.delta_P_matr_alpha.as_mut().unwrap(),
    //                         |&P, &P_prev| P - P_prev,
    //                     );
    //             }
    //         }
    //     }
    //
    //     if is_scf_conv {
    //         println!("{:*<55}", "");
    //         println!("* {:^51} *", "FINAL RESULTS");
    //         println!("{:*<55}", "");
    //         println!("  {:^50}", "RHF SCF (in a.u.)");
    //         println!("  {:=^50}  ", "");
    //         println!("  {:<25}{:>25}", "Total SCF iterations:", scf.tot_scf_iter);
    //         println!("  {:<25}{:>25.18}", "Final SCF energy:", scf.E_scf_conv);
    //         println!("  {:<25}{:>25.18}", "Final tot. energy:", scf.E_tot_conv);
    //         println!("{:*<55}", "");
    //     }
    //     scf
    // }
}

#[inline(always)]
fn scf_conv_check(calc_sett: &CalcSettings, delta_E: f64, rms_comm_val: f64) -> bool {
    (delta_E.abs() < calc_sett.e_diff_thrsh) && (rms_comm_val < calc_sett.commu_conv_thrsh)
}

fn init_diag_F_matr(
    F_matr: &Array2<f64>,
    S_matr_inv_sqrt: &Array2<f64>,
) -> (Array1<f64>, Array2<f64>) {
    let F_matr_pr = S_matr_inv_sqrt.dot(F_matr).dot(S_matr_inv_sqrt);
    let (orb_ener, C_matr_MO) = F_matr_pr.eigh(UPLO::Upper).unwrap();
    (orb_ener, C_matr_MO)
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

    match is_alpha {
        true => {
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
        }
        _ => {
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
}

#[allow(unused)]
fn calc_new_F_matr_dir_scf_uhf(
    F_matr: &mut Array2<f64>,
    delta_P_matr: &Array2<f64>,
    Schwarz_est_int: &Array2<f64>,
    basis: &BasisSet,
) {
    todo!()
}

fn calc_rms_comm_val_uhf(fps_comm_alph: &Array2<f64>, fps_comm_beta: &Array2<f64>) -> f64 {
    let rms_val1 =
        (fps_comm_alph.par_iter().map(|x| x * x).sum::<f64>() / fps_comm_alph.len() as f64).sqrt();
    let rms_val2 =
        (fps_comm_beta.par_iter().map(|x| x * x).sum::<f64>() / fps_comm_beta.len() as f64).sqrt();
    0.5 * (rms_val1 + rms_val2)
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::{calc_type::DiisSettings, print_utils::ExecTimes};
//
//     #[test]
//     fn test_uhf_no_diis_indir_scf() {
//         let mol = Molecule::new("data/xyz/water90.xyz", 0);
//         let basis = BasisSet::new("STO-3G", &mol);
//         let calc_sett = CalcSettings {
//             max_scf_iter: 100,
//             e_diff_thrsh: 1e-8,
//             commu_conv_thrsh: 1e-8,
//             use_diis: false,
//             use_direct_scf: false,
//             diis_sett: DiisSettings {
//                 diis_min: 0,
//                 diis_max: 0,
//             },
//         };
//         let mut exec_times = ExecTimes::new();
//         uhf_scf_normal(&calc_sett, &mut exec_times, &basis, &mol);
//     }
//
//     #[test]
//     fn test_uhf_diis_indir_scf() {
//         let mol = Molecule::new("data/xyz/water90.xyz", 0);
//         let basis = BasisSet::new("STO-3G", &mol);
//         let calc_sett = CalcSettings {
//             max_scf_iter: 100,
//             e_diff_thrsh: 1e-8,
//             commu_conv_thrsh: 1e-8,
//             use_diis: true,
//             use_direct_scf: false,
//             diis_sett: DiisSettings {
//                 diis_min: 2,
//                 diis_max: 6,
//             },
//         };
//         let mut exec_times = ExecTimes::new();
//         uhf_scf_normal(&calc_sett, &mut exec_times, &basis, &mol);
//     }
// }
//
