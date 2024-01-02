#![allow(non_snake_case)]
use crate::calc_type::rhf::calc_cmp_idx;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, Zip};
use std::ops::{Index, IndexMut};

pub(crate) mod rhf;

pub(crate) enum CalcType {
    RHF,
    UHF,
    ROHF,
}

pub struct CalcSettings {
    pub max_scf_iter: usize,
    pub e_diff_thrsh: f64,
    pub rms_p_matr_thrsh: f64,
    pub commu_conv_thrsh: f64,
    pub use_diis: bool,
    pub use_direct_scf: bool,
    pub diis_sett: DiisSettings,
}

#[derive(Debug, Clone, Default)]
pub struct DiisSettings {
    pub diis_min: usize,
    pub diis_max: usize,
}

#[derive(Debug, Clone)]
pub struct EriArr1 {
    eri_arr: Array1<f64>,
}

#[derive(Debug, Default)]
pub struct SCF {
    tot_scf_iter: usize,
    pub E_tot_conv: f64,
    pub E_scf_conv: f64,
    C_matr_conv: Array2<f64>,
    P_matr_conv: Array2<f64>, // [ ] TODO: pot. change this to sparse matrix
    orb_energies_conv: Array1<f64>,
}

#[derive(Debug, Default)]
pub struct DIIS {
    // Better approach
    pub diis_settings: DiisSettings,
    pub F_matr_pr_ring_buf: Vec<Array2<f64>>,
    pub err_matr_pr_ring_buf: Vec<Array2<f64>>,
    // Original approach
    // F_matr_pr_deq: VecDeque<Array2<f64>>,
    // err_matr_pr_deq: VecDeque<Array2<f64>>,
}

impl DIIS {
    pub fn new(diis_settings: &DiisSettings, matr_size: [usize; 2]) -> Self {
        Self {
            diis_settings: diis_settings.clone(),
            F_matr_pr_ring_buf: vec![Array2::<f64>::zeros(matr_size); diis_settings.diis_max],
            err_matr_pr_ring_buf: vec![Array2::<f64>::zeros(matr_size); diis_settings.diis_max],
        }
    }

    #[inline(always)]
    /// This is also the DIIS error matrix
    pub fn calc_FPS_comm(
        F_matr: &Array2<f64>,
        P_matr: &Array2<f64>,
        S_matr: &Array2<f64>,
        // S_matr_inv_sqrt: &Array2<f64>,
    ) -> Array2<f64> {
        F_matr.dot(P_matr).dot(S_matr) - S_matr.dot(P_matr).dot(F_matr)
    }

    pub fn push_to_ring_buf(&mut self, F_matr: &Array2<f64>, err_matr: &Array2<f64>, idx: usize) {
        self.F_matr_pr_ring_buf[idx].assign(F_matr);
        self.err_matr_pr_ring_buf[idx].assign(err_matr);
    }

    // TODO: fix this function
    #[inline]
    fn run_DIIS(&self, error_set_len: usize) -> Array2<f64> {
        let mut B_matr = Array2::<f64>::zeros((error_set_len + 1, error_set_len + 1));
        let mut sol_vec = Array1::<f64>::zeros(error_set_len + 1);
        sol_vec[error_set_len] = -1.0;

        for i in 0..error_set_len {
            for j in 0..=i {
                B_matr[(i, j)] = Zip::from(&self.err_matr_pr_ring_buf[i])
                    .and(&self.err_matr_pr_ring_buf[j])
                    .into_par_iter()
                    .map(|(err_mat1, err_mat2)| err_mat1 * err_mat2)
                    .sum();
                B_matr[(j, i)] = B_matr[(i, j)];
            }
            B_matr[(i, error_set_len)] = -1.0;
            B_matr[(error_set_len, i)] = -1.0;
        }

        println!("B_matr: {:>8.5}", B_matr);

        // * ACTUALLY: Frobenius inner product of matrices (B_ij = error_matr_i * error_matr_j)
        // * OR: flatten error_matr and do dot product
        // Zip::indexed(&mut B_matr).par_for_each(|(idx1, idx2), b_val| {
        //     if idx1 >= idx2 {
        //         *b_val = Zip::from(self.err_matr_pr_ring_buf[idx1])
        //             .and(self.err_matr_pr_ring_buf[idx2])
        //             .into_par_iter()
        //             .map(|(err_mat1, err_mat2)| err_mat1 * err_mat2)
        //             .sum();
        //     }
        // });

        // for i in 0..error_set_len - 1 {
        //     let slice = B_matr.slice(s![i + 1..error_set_len, i]).to_shared();
        //     B_matr.slice_mut(s![i, i + 1..error_set_len]).assign(&slice);
        // }
        //
        // // * Add langrange multiplier to B_matr_extended
        // let new_axis_extension_1 = Array2::from_elem((error_set_len, 1), -1.0_f64);
        // let mut new_axis_extension_2 = Array2::from_elem((1, error_set_len + 1), -1.0_f64);
        // new_axis_extension_2[[0, error_set_len]] = 0.0_f64;
        // let mut B_matr_extended = concatenate![Axis(1), B_matr, new_axis_extension_1];
        // B_matr_extended = concatenate![Axis(0), B_matr_extended, new_axis_extension_2];
        //
        // // * Calculate the coefficients c_vec
        // let c_vec = B_matr_extended.solveh(&sol_vec).unwrap();
        // if is_debug {
        //     println!("c_vec: {:>8.5}", c_vec);
        // }
        //
        // // * Calculate the new DIIS Fock matrix for new D_matr
        let no_cgtos = self.err_matr_pr_ring_buf[0].shape()[0];
        let mut _F_matr_DIIS = Array2::<f64>::zeros((no_cgtos, no_cgtos));
        // for i in 0..error_set_len {
        //     _F_matr_DIIS = _F_matr_DIIS + c_vec[i] * F_matr_set[i];
        // }

        _F_matr_DIIS
    }
}

impl EriArr1 {
    pub fn new(no_bf: usize) -> Self {
        let idx1 = calc_cmp_idx(no_bf, no_bf);
        let eri_max_len = calc_cmp_idx(idx1, idx1) + 1;
        let eri_arr = Array1::<f64>::zeros(eri_max_len);
        Self { eri_arr }
    }
}

impl Index<(usize, usize, usize, usize)> for EriArr1 {
    type Output = f64;
    fn index(&self, idx_tup: (usize, usize, usize, usize)) -> &f64 {
        let cmp_idx1 = calc_cmp_idx(idx_tup.0, idx_tup.1);
        let cmp_idx2 = calc_cmp_idx(idx_tup.2, idx_tup.3);
        let cmp_idx = calc_cmp_idx(cmp_idx1, cmp_idx2);

        &self.eri_arr[cmp_idx]
    }
}

/// This is for the case, where the cmp_idx is already computed
impl Index<usize> for EriArr1 {
    type Output = f64;
    fn index(&self, cmp_idx: usize) -> &f64 {
        &self.eri_arr[cmp_idx]
    }
}

impl IndexMut<(usize, usize, usize, usize)> for EriArr1 {
    fn index_mut(&mut self, idx_tup: (usize, usize, usize, usize)) -> &mut f64 {
        let cmp_idx1 = calc_cmp_idx(idx_tup.0, idx_tup.1);
        let cmp_idx2 = calc_cmp_idx(idx_tup.2, idx_tup.3);
        let cmp_idx = calc_cmp_idx(cmp_idx1, cmp_idx2);

        &mut self.eri_arr[cmp_idx]
    }
}

/// This is for the case, where the cmp_idx is already computed
impl IndexMut<usize> for EriArr1 {
    fn index_mut(&mut self, cmp_idx: usize) -> &mut f64 {
        &mut self.eri_arr[cmp_idx]
    }
}
