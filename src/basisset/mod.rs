mod parser;

// use crate::molecule::atom::Atom;
use ndarray::Array1;


/// # Basis set 
/// ## Arguments 
/// * `name` - name of the basis set
/// * `no_ao` - number of atomic orbitals
/// * `no_bf` - number of basis functions
/// * `shells` - vector of shells (A shell is a collection of CGTOs with the same angular momentum)
/// * `use_pure_am` - whether to use pure angular momentum (true) or cartesian (false)
/// 
/// ## Notes
/// * `no_bf` = `no_ao` * 2 if UHF; `no_bf` = `no_ao` if RHF
#[derive(Debug)]
struct BasisSet {
    name: String,
    no_ao: usize, 
    no_bf: usize, 
    shells: Vec<Shell>,    
    use_pure_am: bool, 
    // atoms: Vec<Atom>,
    // bfs: Vec<Bf>,
}

#[derive(Debug)]
struct Shell {
    ang_mom: i32, 
    is_pure_am: bool,
}

// impl BasisSet {
//     fn new(name: String, no_ao: usize, no_bf: usize, shells: Vec<Shell>, use_pure_am: bool) -> Self { Self { name, no_ao, no_bf, shells, use_pure_am } }
// }

#[derive(Debug)]
struct CGTO {
    pgto_vec: Vec<PGTO>,
    no_pgtos: usize,
}

#[derive(Clone, Debug)]
struct PGTO {
    alpha: f64,
    pgto_coeff: f64,
    gauss_center_pos: Array1<f64>,
    ang_mom_vec: [i32; 3],
    norm_const: f64,
}

// struct Bf {
//     atom: usize,
//     l: usize,
//     m: usize,
//     n: usize,
//     zeta: f64,
//     c: Vec<f64>,
// }



#[derive(Debug, Default)]
pub struct BasisSetTotal {
    pub basis_set_cgtos: Vec<CGTO>,
    pub no_cgtos: usize,
    pub no_occ_orb: usize,
    pub center_charge: Array1<f64>,
    pub dipole_moment_total: Array1<f64>,
}
