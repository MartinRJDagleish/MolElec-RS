#![allow(non_snake_case)]
use crate::calc_type::rhf::calc_cmp_idx;
use ndarray::{Array1, Array2};
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
    pub commu_conv_thrsh: f64,
    pub use_diis: bool,
    pub use_direct_scf: bool,
    pub diis_sett: DiisSettings,
}

pub struct DiisSettings {
    pub diis_start: usize,
    pub diis_end: usize,
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
struct DIIS {
    // Better approach
    F_matr_pr_ring_buf: Vec<Array2<f64>>,
    err_matr_pr_ring_buf: Vec<Array2<f64>>,
    // Original approach
    // F_matr_pr_deq: VecDeque<Array2<f64>>,
    // err_matr_pr_deq: VecDeque<Array2<f64>>,
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


