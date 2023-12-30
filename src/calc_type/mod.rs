#![allow(non_snake_case)]
use crate::calc_type::rhf::calc_cmp_idx;
use ndarray::{Array1, Array2};
use std::ops::{Index, IndexMut};

mod rhf;

pub struct CalcSettings {
    pub max_scf_iter: usize,
    pub e_diff_thrsh: f64,
    pub commu_conv_thrsh: f64,
    pub use_diis: bool,
    pub diis_sett: DiisSettings,
}

pub struct DiisSettings {
    pub diis_start: usize,
    pub diis_end: usize,
    pub diis_max: usize,
}

#[derive(Debug, Clone)]
pub struct ERI_Arr1 {
    eri_arr: Array1<f64>,
}

impl ERI_Arr1 {
    pub fn new(no_bf: usize) -> Self {
        let eri_arr = Array1::<f64>::zeros(no_bf * no_bf * no_bf * no_bf);
        Self { eri_arr }
    }
}

impl Index<(usize, usize, usize, usize)> for ERI_Arr1 {
    type Output = f64; 
    fn index(&self, idx_tup: (usize, usize, usize, usize)) -> &f64 {
        let cmp_idx1 = calc_cmp_idx(idx_tup.0, idx_tup.1);
        let cmp_idx2 = calc_cmp_idx(idx_tup.2, idx_tup.3);
        let cmp_idx = calc_cmp_idx(cmp_idx1, cmp_idx2);

        &self.eri_arr[cmp_idx]
    }
}

/// This is for the case, where the cmp_idx is already computed
impl Index<usize> for ERI_Arr1 {
    type Output = f64; 
    fn index(&self, cmp_idx: usize) -> &f64 {
        &self.eri_arr[cmp_idx]
    }
}

impl IndexMut<(usize, usize, usize, usize)> for ERI_Arr1 {
    fn index_mut(&mut self, idx_tup: (usize, usize, usize, usize)) -> &mut f64 {
        let cmp_idx1 = calc_cmp_idx(idx_tup.0, idx_tup.1);
        let cmp_idx2 = calc_cmp_idx(idx_tup.2, idx_tup.3);
        let cmp_idx = calc_cmp_idx(cmp_idx1, cmp_idx2);

        &mut self.eri_arr[cmp_idx]
    }
}

/// This is for the case, where the cmp_idx is already computed
impl IndexMut<usize> for ERI_Arr1 {
    fn index_mut(&mut self, cmp_idx: usize) -> &mut f64 {
        &mut self.eri_arr[cmp_idx]
    }
}

// impl IndexMut<usize> for ERI {
//     fn index_mut(&mut self, i: usize) -> &mut f64 {
//         match i {
//             0 => &mut self.x,
//             1 => &mut self.y,
//             2 => &mut self.z,
//             _ => panic!("Index out of bounds for Atom"),
//         }
//     }
// }

#[derive(Debug, Default)]
struct SCF {
    tot_scf_iter: usize,
    E_tot: f64,
    E_scf: f64,
    C_matr_final: Array2<f64>,
    P_matr_final: Array2<f64>, // [ ] TODO: pot. change this to sparse matrix
    orb_energies_final: Vec<f64>,
    diis: DIIS,
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

impl DIIS {
    fn new() -> DIIS {
        DIIS {
            F_matr_pr_ring_buf: Vec::<Array2<f64>>::new(),
            err_matr_pr_ring_buf: Vec::<Array2<f64>>::new(),
            // F_matr_pr_deq: VecDeque::<Array2<f64>>::new(),
            // err_matr_pr_deq: VecDeque::<Array2<f64>>::new(),
        }
    }
}

impl SCF {
    fn new() -> SCF {
        let diis = DIIS::new();
        SCF {
            tot_scf_iter: 0,
            E_tot: 0.0_f64,
            E_scf: 0.0_f64,
            C_matr_final: Array2::<f64>::zeros((1, 1)),
            P_matr_final: Array2::<f64>::zeros((1, 1)),
            orb_energies_final: Vec::<f64>::new(),
            diis,
        }
    }
}
