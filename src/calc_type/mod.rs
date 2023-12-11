#![allow(non_snake_case)]
use ndarray::Array2;

mod rhf;

#[derive(Debug,Default)]
struct SCF {
    tot_scf_iter: usize,    
    E_tot: f64,
    E_scf: f64,
    C_matr_final: Array2<f64>,
    P_matr_final: Array2<f64>, //TODO: pot. change this to sparse matrix
    orb_energies_final: Vec<f64>,     
    diis: DIIS,
}

#[derive(Debug,Default)]
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
            C_matr_final: Array2::<f64>::zeros((1,1)),
            P_matr_final: Array2::<f64>::zeros((1,1)),
            orb_energies_final: Vec::<f64>::new(),
            diis,
        }
    }
    
}