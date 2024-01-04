#![allow(non_snake_case)]
use crate::calc_type::rhf::calc_cmp_idx;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, Zip};
use ndarray_linalg::SolveH;
use std::ops::{Index, IndexMut};

pub(crate) mod rhf;
pub(crate) mod uhf;

pub(crate) enum CalcType {
    RHF,
    UHF,
    ROHF,
}

pub struct CalcSettings {
    pub max_scf_iter: usize,
    pub e_diff_thrsh: f64,
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
    C_matr_conv_alph: Array2<f64>,
    P_matr_conv_alph: Array2<f64>, // [ ] TODO: pot. change this to sparse matrix
    C_matr_conv_beta: Option<Array2<f64>>,
    P_matr_conv_beta: Option<Array2<f64>>, // [ ] TODO: pot. change this to sparse matrix
    orb_E_conv_alph: Array1<f64>,
    orb_E_conv_beta: Option<Array1<f64>>
}

#[derive(Debug, Default,Clone)]
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
    ) -> Array2<f64> {
        F_matr.dot(P_matr).dot(S_matr) - S_matr.dot(P_matr).dot(F_matr)
    }

    pub fn push_to_ring_buf(&mut self, F_matr: &Array2<f64>, err_matr: &Array2<f64>, idx: usize) {
        self.F_matr_pr_ring_buf[idx].assign(F_matr);
        self.err_matr_pr_ring_buf[idx].assign(err_matr);
    }

    /// - Source: Pulay DIIS paper (1980)
    /// - Link: https://doi.org/10.1016/0009-2614(80)80396-4
    /// - Source: Pulay DIIS improvment paper (1982)
    /// - Link: https://doi.org/10.1002/jcc.540030413
    /// → using e' = A^+(FPS - SPF)A here
    ///
    /// see also: https://en.wikipedia.org/wiki/DIIS
    /// TODO: only recalculate the last row and column of B_matr
    fn run_DIIS(&self, error_set_len: usize) -> Array2<f64> {
        let mut B_matr = Array2::<f64>::zeros((error_set_len + 1, error_set_len + 1));
        let mut sol_vec = Array1::<f64>::zeros(error_set_len + 1);
        sol_vec[error_set_len] = -1.0;

        for i in 0..error_set_len {
            for j in 0..=i {
                B_matr[(i, j)] = Zip::from(&self.err_matr_pr_ring_buf[i])
                    .and(&self.err_matr_pr_ring_buf[j])
                    .into_par_iter()
                    .map(|(err_mat1, err_mat2)| *err_mat1 * *err_mat2)
                    .sum::<f64>();
                B_matr[(j, i)] = B_matr[(i, j)];
            }
            B_matr[(i, error_set_len)] = -1.0;
            B_matr[(error_set_len, i)] = -1.0;
        }

        // Calculate the coefficients c_vec
        let c_vec = B_matr.solveh(&sol_vec).unwrap();
        
        // Calculate the new DIIS Fock matrix for new D_matr
        let no_cgtos = self.err_matr_pr_ring_buf[0].shape()[0];
        let mut _F_matr_DIIS = Array2::<f64>::zeros((no_cgtos, no_cgtos));
        for i in 0..error_set_len {
            _F_matr_DIIS = _F_matr_DIIS + c_vec[i] * &self.F_matr_pr_ring_buf[i];
        }

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
