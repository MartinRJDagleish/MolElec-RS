use ndarray::Array2;
use ndarray_linalg::{InverseH, Eigh, UPLO};

use crate::{
    basisset::BasisSet,
    calc_type::{HF_Ref, SCF, rhf::{RHF, matr_inv_ssqrt}, HF},
    molecule::Molecule,
    print_utils::{print_scf::print_scf_header_and_settings, ExecTimes, fmt_f64}, mol_int_and_deriv::te_int::calc_schwarz_est_int,
};

use super::CalcSettings;

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
            let S_matr_inv_sqrt = matr_inv_ssqrt(&S_matr, UPLO::Upper);
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
    
    #[test]
    fn test_calc_energy_gradient_p_matr() {
        let F_matr = Array2::<f64>::zeros((3, 3));
        let P_matr = Array2::<f64>::zeros((3, 3));
        let S_matr = Array2::<f64>::zeros((3, 3));

        let grad = calc_energy_gradient_p_matr(&F_matr, &P_matr, &S_matr);
        assert_eq!(grad, Array2::<f64>::zeros((3, 3)));
    }
}
