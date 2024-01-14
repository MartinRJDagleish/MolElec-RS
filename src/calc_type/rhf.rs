use super::{CalcSettings, HFMatrices, SCF, HF};
use crate::{
    basisset::BasisSet,
    calc_type::{EriArr1, HF_Ref, DIIS},
    mol_int_and_deriv::{
        oe_int::{calc_kinetic_int_cgto, calc_overlap_int_cgto, calc_pot_int_cgto},
        te_int::{calc_ERI_int_cgto, calc_schwarz_est_int, calc_schwarz_est_int_inp},
    },
    molecule::Molecule,
    print_utils::{fmt_f64, print_scf::print_scf_header_and_settings, ExecTimes},
};

use ndarray::{linalg::general_mat_mul, parallel::prelude::*, prelude::*, Zip};
use ndarray_linalg::{Eigh, InverseH, UPLO};

#[derive(Debug, Clone, Default)]
pub(crate) struct RHF {
    // Matrices needed for the SCF calculation
    hf_matrs: HFMatrices,

    // f64 values for SCF calculation
    E_scf_prev: f64,
    E_scf_curr: f64,
    E_tot_prev: f64,
    E_tot_curr: f64,
}

impl HF for RHF {
    /// This is the main RHF SCF function to be called and run the RHF SCF calculation
    /// ## Options:
    /// - DIIS
    /// - direct vs. indirect SCF
    #[allow(unused)]
    fn run_scf(
        &mut self,
        calc_sett: &CalcSettings,
        exec_times: &mut ExecTimes,
        basis: &BasisSet,
        mol: &Molecule,
    ) -> SCF {
        print_scf_header_and_settings(calc_sett, HF_Ref::RHF_ref);

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

        // Initial guess -> H_core
        // TODO: [ ] replace with guess
        self.hf_matrs.F_matr_alpha = self.hf_matrs.H_core_matr.clone();

        // Print SCF iteration Header
        println!(
            "{:>3} {:^20} {:^20} {:^20} {:^20}",
            "Iter", "E_scf", "E_tot", "ΔE", "RMS(|FPS - SPF|)"
        );
        let mut diis_str = "";
        for scf_iter in 0..=calc_sett.max_scf_iter {
            if scf_iter == 0 {
                self.hf_matrs.F_matr_pr_alpha = self
                    .hf_matrs
                    .S_matr_inv_sqrt
                    .dot(&self.hf_matrs.F_matr_alpha)
                    .dot(&self.hf_matrs.S_matr_inv_sqrt);

                (self.hf_matrs.orb_ener_alpha, self.hf_matrs.C_matr_MO_alpha) =
                    self.hf_matrs.F_matr_pr_alpha.eigh(UPLO::Upper).unwrap();
                self.hf_matrs.C_matr_AO_alpha = self
                    .hf_matrs
                    .S_matr_inv_sqrt
                    .dot(&self.hf_matrs.C_matr_MO_alpha);

                Self::calc_P_matr_rhf(
                    &mut self.hf_matrs.P_matr_alpha,
                    &self.hf_matrs.C_matr_AO_alpha,
                    basis.no_occ(),
                );
                if calc_sett.use_direct_scf {
                    self.hf_matrs.delta_P_matr_alpha = Some(self.hf_matrs.P_matr_alpha.clone());
                }
            } else {
                /// direct or indirect scf
                match self.hf_matrs.eri_opt {
                    Some(ref eri) => {
                        Self::calc_new_F_matr_ind_scf_rhf(
                            &mut self.hf_matrs.F_matr_alpha,
                            &self.hf_matrs.H_core_matr,
                            &self.hf_matrs.P_matr_alpha,
                            eri,
                        );
                    }
                    None => {
                        Self::calc_new_F_matr_dir_scf_rhf(
                            &mut self.hf_matrs.F_matr_alpha,
                            self.hf_matrs.delta_P_matr_alpha.as_ref().unwrap(),
                            self.hf_matrs.schwarz_est.as_ref().unwrap(),
                            basis,
                        );
                    }
                }
                self.E_scf_curr = Self::calc_E_scf_rhf(
                    &self.hf_matrs.P_matr_alpha,
                    &self.hf_matrs.H_core_matr,
                    &self.hf_matrs.F_matr_alpha,
                );
                self.E_tot_curr = self.E_scf_curr + V_nuc;
                // FPS - SPF
                let fps_comm = DIIS::calc_FPS_comm(
                    &self.hf_matrs.F_matr_alpha,
                    &self.hf_matrs.P_matr_alpha,
                    &self.hf_matrs.S_matr,
                );

                // F' = S^(-1/2) * F * S^(-1/2)
                self.hf_matrs.F_matr_pr_alpha = self
                    .hf_matrs
                    .S_matr_inv_sqrt
                    .dot(&self.hf_matrs.F_matr_alpha)
                    .dot(&self.hf_matrs.S_matr_inv_sqrt);

                if calc_sett.use_diis {
                    let repl_idx = (scf_iter - 1) % calc_sett.diis_sett.diis_max; // always start with 0
                    let err_matr = self
                        .hf_matrs
                        .S_matr_inv_sqrt
                        .dot(&fps_comm)
                        .dot(&self.hf_matrs.S_matr_inv_sqrt);
                    diis.as_mut().unwrap().push_to_ring_buf(
                        &self.hf_matrs.F_matr_pr_alpha,
                        &err_matr,
                        repl_idx,
                    );

                    if scf_iter >= calc_sett.diis_sett.diis_min {
                        let err_set_len = std::cmp::min(calc_sett.diis_sett.diis_max, scf_iter);
                        self.hf_matrs.F_matr_pr_alpha =
                            diis.as_ref().unwrap().run_DIIS(err_set_len);
                        diis_str = "DIIS";
                    }
                }

                (self.hf_matrs.orb_ener_alpha, self.hf_matrs.C_matr_MO_alpha) =
                    self.hf_matrs.F_matr_pr_alpha.eigh(UPLO::Upper).unwrap();
                self.hf_matrs.C_matr_AO_alpha = self
                    .hf_matrs
                    .S_matr_inv_sqrt
                    .dot(&self.hf_matrs.C_matr_MO_alpha);

                let delta_E = self.E_scf_curr - self.E_scf_prev;
                let rms_comm_val = (fps_comm.par_iter().map(|x| x * x).sum::<f64>()
                    / fps_comm.len() as f64)
                    .sqrt();
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

                if (delta_E.abs() < calc_sett.e_diff_thrsh)
                    && (rms_comm_val < calc_sett.commu_conv_thrsh)
                {
                    scf.tot_scf_iter = scf_iter;
                    scf.E_scf_conv = self.E_scf_curr;
                    scf.E_tot_conv = self.E_tot_curr;
                    scf.C_matr_conv_alph = self.hf_matrs.C_matr_AO_alpha.clone();
                    scf.P_matr_conv_alph = self.hf_matrs.P_matr_alpha.clone();
                    println!("P_matr conv:\n{:>12.8}", &self.hf_matrs.P_matr_alpha);
                    scf.C_matr_conv_beta = None;
                    scf.P_matr_conv_beta = None;
                    scf.orb_E_conv_alph = self.hf_matrs.orb_ener_alpha.clone();
                    println!("\nSCF CONVERGED!\n");
                    is_scf_conv = true;
                    break;
                } else if scf_iter == calc_sett.max_scf_iter {
                    println!("\nSCF DID NOT CONVERGE!\n");
                    break;
                }
                self.E_scf_prev = self.E_scf_curr;
                self.hf_matrs.P_matr_prev_alpha = self.hf_matrs.P_matr_alpha.clone();
                Self::calc_P_matr_rhf(
                    &mut self.hf_matrs.P_matr_alpha,
                    &self.hf_matrs.C_matr_AO_alpha,
                    basis.no_occ(),
                );
                if calc_sett.use_direct_scf {
                    Zip::from(&self.hf_matrs.P_matr_alpha.view())
                        .and(&self.hf_matrs.P_matr_prev_alpha.view())
                        .par_map_assign_into(
                            self.hf_matrs.delta_P_matr_alpha.as_mut().unwrap(),
                            |&P, &P_prev| P - P_prev,
                        );
                }
            }
        }

        if is_scf_conv {
            println!("{:*<55}", "");
            println!("* {:^51} *", "FINAL RESULTS");
            println!("{:*<55}", "");
            println!("  {:^50}", "RHF SCF (in a.u.)");
            println!("  {:=^50}  ", "");
            println!("  {:<25}{:>25}", "Total SCF iterations:", scf.tot_scf_iter);
            println!("  {:<25}{:>25.18}", "Final SCF energy:", scf.E_scf_conv);
            println!("  {:<25}{:>25.18}", "Final tot. energy:", scf.E_tot_conv);
            println!("{:*<55}", "");
        }
        scf
    }
}

impl RHF {
    pub fn new(basis: &BasisSet, calc_sett: &CalcSettings, ref_type: HF_Ref) -> Self {
        let create_beta_vars = match ref_type {
            HF_Ref::RHF_ref => false,
            HF_Ref::UHF_ref | HF_Ref::ROHF_ref => true,
        };

        Self {
            hf_matrs: HFMatrices::new(basis.no_bf(), calc_sett.use_direct_scf, create_beta_vars),
            E_scf_prev: 0.0,
            E_scf_curr: 0.0,
            E_tot_prev: 0.0,
            E_tot_curr: 0.0,
        }
    }

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

        // Return ovelap and core hamiltonian (T + V)
        (S_matr, T_matr + V_matr)
    }

    pub fn calc_2e_int_matr(basis: &BasisSet) -> EriArr1 {
        let mut eri = EriArr1::new(basis.no_bf());

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
                                                    eri[cmp_idx] = calc_ERI_int_cgto(
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

        eri
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
        println!("S_matr: \n{:>12.8}", &self.hf_matrs.S_matr);
        println!("T_matr: \n{:>12.8}", &self.hf_matrs.T_matr);
        println!("V_matr: \n{:>12.8}", &self.hf_matrs.V_ne_matr);

        Zip::from(self.hf_matrs.T_matr.view())
            .and(self.hf_matrs.V_ne_matr.view())
            .par_map_assign_into(&mut self.hf_matrs.H_core_matr, |&t, &v| t + v);
        println!("FINSIHED calculating 1e integrals ...");

        println!("Starting orthogonalization matrix calculation ...");
        self.hf_matrs.S_matr_inv_sqrt = Self::inv_ssqrt(&self.hf_matrs.S_matr, UPLO::Upper);
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


    fn calc_P_matr_rhf(P_matr: &mut Array2<f64>, C_matr: &Array2<f64>, no_occ: usize) {
        let C_occ = C_matr.slice(s![.., ..no_occ]);
        general_mat_mul(2.0_f64, &C_occ, &C_occ.t(), 0.0_f64, P_matr)
    }

    fn calc_new_F_matr_ind_scf_rhf(
        F_matr: &mut Array2<f64>,
        H_core: &Array2<f64>,
        P_matr: &Array2<f64>,
        eri: &EriArr1,
    ) {
        F_matr.assign(H_core);
        let no_bf = F_matr.nrows();

        for mu in 0..no_bf {
            for nu in 0..=mu {
                for lambda in 0..no_bf {
                    for sigma in 0..no_bf {
                        F_matr[(mu, nu)] += P_matr[(lambda, sigma)]
                            * (eri[(mu, nu, lambda, sigma)] - 0.5 * eri[(mu, sigma, lambda, nu)]);
                    }
                }
                F_matr[(nu, mu)] = F_matr[(mu, nu)];
            }
        }
    }

    /// Calc Fock matrix in a direct SCF fashion (i.e. without precomputed eri tensor)
    fn calc_new_F_matr_dir_scf_rhf(
        F_matr: &mut Array2<f64>,
        delta_P_matr: &Array2<f64>,
        Schwarz_est_int: &Array2<f64>,
        basis: &BasisSet,
    ) {
        const INT_THRSH: f64 = 1e-12;
        let mut G_matr = Array2::<f64>::zeros((basis.no_bf(), basis.no_bf()));

        for (sh_idx1, shell1) in basis.shell_iter().enumerate() {
            for (cgto_idx1, cgto1) in shell1.cgto_iter().enumerate() {
                let mu = basis.sh_len_offset(sh_idx1) + cgto_idx1;

                for (sh_idx2, shell2) in basis.shell_iter().enumerate() {
                    for (cgto_idx2, cgto2) in shell2.cgto_iter().enumerate() {
                        let nu = basis.sh_len_offset(sh_idx2) + cgto_idx2;
                        let max_dP_val1 = 2.0 * delta_P_matr[(mu, nu)].abs();

                        for (sh_idx3, shell3) in basis.shell_iter().enumerate() {
                            for (cgto_idx3, cgto3) in shell3.cgto_iter().enumerate() {
                                let lam = basis.sh_len_offset(sh_idx3) + cgto_idx3;
                                let max_dP_val4 = 0.5 * delta_P_matr[(nu, lam)].abs();
                                let max_dP_val5 = 0.5 * delta_P_matr[(mu, lam)].abs();
                                let max_dP_first_three = [max_dP_val1, max_dP_val4, max_dP_val5]
                                    .iter()
                                    .fold(f64::MIN, |a, &b| a.max(b));

                                for (sh_idx4, shell4) in basis.shell_iter().enumerate() {
                                    for (cgto_idx4, cgto4) in shell4.cgto_iter().enumerate() {
                                        let sig = basis.sh_len_offset(sh_idx4) + cgto_idx4;
                                        let max_dP_val2 = 2.0 * delta_P_matr[(lam, sig)].abs();
                                        let max_dP_val3 = 0.5 * delta_P_matr[(mu, sig)].abs();
                                        let max_dP_val6 = 0.5 * delta_P_matr[(nu, sig)].abs();

                                        let max_dP_val = [
                                            max_dP_first_three,
                                            max_dP_val2,
                                            max_dP_val3,
                                            max_dP_val6,
                                        ]
                                        .iter()
                                        .fold(f64::MIN, |a, &b| a.max(b));

                                        let int_est = Schwarz_est_int[(mu, nu)]
                                            * Schwarz_est_int[(lam, sig)]
                                            * max_dP_val;
                                        if int_est >= INT_THRSH {
                                            let eri_val =
                                                0.5 * calc_ERI_int_cgto(cgto1, cgto2, cgto3, cgto4);

                                            // 1. Coulomb type contribution
                                            G_matr[(mu, nu)] += delta_P_matr[(lam, sig)] * eri_val;
                                            G_matr[(lam, sig)] += delta_P_matr[(mu, nu)] * eri_val;

                                            // 2. Exchange type contribution
                                            G_matr[(mu, lam)] -=
                                                0.25_f64 * delta_P_matr[(nu, sig)] * eri_val;
                                            G_matr[(nu, sig)] -=
                                                0.25_f64 * delta_P_matr[(mu, lam)] * eri_val;
                                            G_matr[(mu, sig)] -=
                                                0.25_f64 * delta_P_matr[(nu, lam)] * eri_val;
                                            G_matr[(nu, lam)] -=
                                                0.25_f64 * delta_P_matr[(mu, sig)] * eri_val;
                                        } else {
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let GG_matr = &G_matr + &G_matr.t();
        Zip::from(F_matr)
            .and(&GG_matr)
            .into_par_iter()
            .for_each(|(f, gg)| *f += 0.5 * *gg);
    }

    fn calc_E_scf_rhf(P_matr: &Array2<f64>, H_core: &Array2<f64>, F_matr: &Array2<f64>) -> f64 {
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

    pub(crate) fn inv_ssqrt(arr2: &Array2<f64>, uplo: UPLO) -> Array2<f64> {
        let (e, v) = arr2.eigh(uplo).unwrap();
        let e_inv_sqrt = Array1::from_iter(e.iter().map(|x| x.powf(-0.5)));
        let e_inv_sqrt_diag = Array::from_diag(&e_inv_sqrt);
        let result = v.dot(&e_inv_sqrt_diag).dot(&v.t());
        result
    }
}

#[inline(always)]
pub fn calc_cmp_idx(idx1: usize, idx2: usize) -> usize {
    if idx1 >= idx2 {
        idx1 * (idx1 + 1) / 2 + idx2
    } else {
        idx2 * (idx2 + 1) / 2 + idx1
    }
}
/// My try of a implementation of the density matrix based approach to solve the RHF SCF equations.
/// This is NOT actually linear scaling, since I am not using any tricks to reduce the computational cost.
/// No sparse matrices or anything like that.
///
/// Missing: CFMM for Coulomb and LinK for Exchange matrix
#[allow(unused)]
fn rhf_scf_linscal(
    calc_sett: &CalcSettings,
    exec_times: &mut ExecTimes,
    basis: &BasisSet,
    mol: &Molecule,
) {
    print_scf_header_and_settings(calc_sett, HF_Ref::RHF_ref);
    const SHOW_ALL_CONV_CRIT: bool = false;

    let mut is_scf_conv = false;
    let mut scf = SCF::default();

    let V_nuc: f64 = if mol.no_atoms() > 100 {
        mol.calc_core_potential_par()
    } else {
        mol.calc_core_potential_ser()
    };

    println!("Calculating 1e integrals ...");
    let (S_matr, H_core) = RHF::calc_1e_int_matrs(basis, mol);
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
        eri = Some(RHF::calc_2e_int_matr(basis));
        println!("FINSIHED calculating 2e integrals ...");
    }

    // Init matrices for SCF loop
    let mut E_scf_prev = 0.0;

    let mut P_matr = Array2::<f64>::zeros((basis.no_bf(), basis.no_bf()));
    let mut P_matr_old = P_matr.clone();
    let mut delta_P_matr = P_matr.clone();
    let mut diis_str = "";

    // Initial guess -> H_core
    let mut F_matr = H_core.clone();

    let S_matr_inv = S_matr.invh().unwrap();

    // Print SCF iteration Header
    println!(
        "{:>3} {:^20} {:^20} {:^20} {:^20}",
        "Iter", "E_scf", "E_tot", "ΔE", "RMS(|FPS - SPF|)"
    );
    for scf_iter in 0..=calc_sett.max_scf_iter {
        if scf_iter == 0 {
            let S_matr_inv_sqrt = RHF::inv_ssqrt(&S_matr, UPLO::Upper);
            let F_matr_pr = S_matr_inv_sqrt.dot(&F_matr).dot(&S_matr_inv_sqrt);

            let (orb_ener, C_matr_MO) = F_matr_pr.eigh(UPLO::Upper).unwrap();
            let C_matr_AO = S_matr_inv_sqrt.dot(&C_matr_MO);

            RHF::calc_P_matr_rhf(&mut P_matr, &C_matr_AO, basis.no_occ());
            println!("Guess P_matr:\n {:12.6}", &P_matr);
            println!(
                "Eigenvalues of guess: {:12.6}",
                &P_matr.eigh(UPLO::Upper).unwrap().0
            );
            delta_P_matr = P_matr.clone();
        } else {
            // 1. First new Fock matrix
            RHF::calc_new_F_matr_dir_scf_rhf(
                &mut F_matr,
                &delta_P_matr,
                schwarz_est_matr.as_ref().unwrap(),
                basis,
            );
            // 2. Calc E_scf
            let E_scf_curr = RHF::calc_E_scf_rhf(&P_matr, &H_core, &F_matr);
            scf.E_tot_conv = E_scf_curr + V_nuc;

            let delta_E = E_scf_curr - E_scf_prev;
            println!(
                "{:>3} {:>20.12} {:>20.12} {}",
                scf_iter,
                E_scf_curr,
                scf.E_tot_conv,
                fmt_f64(delta_E, 20, 8, 2),
            );

            // 3. Save old density matrix
            P_matr_old = P_matr.clone();
            // 4. Calculate new density matrix
            P_matr = calc_new_P_matr_linscal_contravar(&P_matr, &F_matr, &S_matr, &S_matr_inv);
            // 5. Calculate calculate ΔP matrix
            delta_P_matr = (&P_matr - &P_matr_old).to_owned();
            E_scf_prev = E_scf_curr;
        }
    }
}

#[allow(unused)]
/// Result is contravariant
fn calc_new_P_matr_linscal_contravar(
    P_matr_curr: &Array2<f64>,
    F_matr: &Array2<f64>,
    S_matr: &Array2<f64>,
    S_matr_inv: &Array2<f64>,
) -> Array2<f64> {
    const STEP_WIDTH: f64 = 0.000_001;

    let energy_grad = calc_energy_gradient_p_matr(F_matr, P_matr_curr, S_matr);
    P_matr_curr - STEP_WIDTH * S_matr_inv.dot(&energy_grad).dot(S_matr_inv)
}

fn calc_energy_gradient_p_matr(
    F_matr: &Array2<f64>,
    P_matr: &Array2<f64>,
    S_matr: &Array2<f64>,
) -> Array2<f64> {
    3.0 * F_matr.dot(P_matr).dot(S_matr) + 3.0 * S_matr.dot(P_matr).dot(F_matr)
        - 2.0 * S_matr.dot(P_matr).dot(F_matr).dot(P_matr).dot(S_matr)
        - 2.0 * F_matr.dot(P_matr).dot(S_matr).dot(P_matr).dot(S_matr)
        - 2.0 * S_matr.dot(P_matr).dot(S_matr).dot(P_matr).dot(F_matr)
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
        // let rhf = RHF::default();

        let (S_matr, H_core) = RHF::calc_1e_int_matrs(&basis, &mol);
        println!("{}", S_matr);
        println!("{}", H_core);
    }

    #[test]
    fn test_calc_2e_int_matr() {
        println!("Test calc_2e_int_matr");
        let mol = Molecule::new("data/xyz/water90.xyz", 0);
        let basis = BasisSet::new("STO-3G", &mol);

        let eri = RHF::calc_2e_int_matr(&basis);
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

        let mut rhf = RHF::new(&basis, &calc_sett, HF_Ref::RHF_ref);
        let _scf = rhf.run_scf(&calc_sett, &mut exec_times, &basis, &mol);
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

        let mut rhf = RHF::new(&basis, &calc_sett, HF_Ref::RHF_ref);
        let _scf = rhf.run_scf(&calc_sett, &mut exec_times, &basis, &mol);
        // println!("{:?}", _scf);
    }

    #[test]
    fn test_rhf_dir_diis() {
        let mol = Molecule::new("data/xyz/water90.xyz", 0);
        let basis = BasisSet::new("STO-3G", &mol);
        let calc_sett = CalcSettings {
            max_scf_iter: 100,
            e_diff_thrsh: 1e-8,
            commu_conv_thrsh: 1e-6,
            use_diis: true,
            use_direct_scf: true,
            diis_sett: DiisSettings {
                diis_min: 2,
                diis_max: 6,
            },
        };
        let mut exec_times = ExecTimes::new();

        let mut rhf = RHF::new(&basis, &calc_sett, HF_Ref::RHF_ref);
        let _scf = rhf.run_scf(&calc_sett, &mut exec_times, &basis, &mol);
        // println!("{:?}", _scf);
    }

    #[test]
    fn test_rhf_indir_diis_sad() {
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

        let mut rhf = RHF::new(&basis, &calc_sett, HF_Ref::RHF_ref);
        let _scf = rhf.run_scf(&calc_sett, &mut exec_times, &basis, &mol);
    }

    // #[test]
    // fn test_rhf_dir_linscal() {
    //     let mol = Molecule::new("data/xyz/water90.xyz", 0);
    //     let basis = BasisSet::new("STO-3G", &mol);
    //     let calc_sett = CalcSettings {
    //         max_scf_iter: 100,
    //         e_diff_thrsh: 1e-8,
    //         commu_conv_thrsh: 1e-6,
    //         use_diis: false,
    //         use_direct_scf: true,
    //         diis_sett: DiisSettings {
    //             diis_min: 2,
    //             diis_max: 6,
    //         },
    //     };
    //     let mut exec_times = ExecTimes::new();
    //
    //     let _scf = rhf_scf_linscal(calc_sett, &mut exec_times, &basis, &mol);
    // }
}
