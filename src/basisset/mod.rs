mod parser;

use crate::molecule::{atom::Atom, Molecule};
use ndarray::Array1;
use parser::BasisSetDefTotal;

/// # Basis set
/// ## Arguments
/// - `name` - name of the basis set
/// - `no_ao` - number of atomic orbitals
/// - `no_bf` - number of basis functions
/// - `shells` - vector of shells (A shell is a collection of CGTOs with the same angular momentum)
/// - `use_pure_am` - whether to use pure angular momentum (true) or cartesian (false)
///
/// ## Notes
/// - `no_bf` = `no_ao` * 2 if UHF; `no_bf` = `no_ao` if RHF
#[derive(Debug)]
pub(crate) struct BasisSet<'a> {
    name: String,
    use_pure_am: bool,
    no_ao: usize,
    no_bf: usize,
    shells: Vec<Shell<'a>>,
}

#[derive(Debug)]
struct Shell<'a> {
    ang_mom: i32,
    is_pure_am: bool,
    cgtos: Vec<CGTO<'a>>, // == basis funcs.
    center_pos: &'a Atom,
}

#[derive(Debug)]
struct CGTO<'a> {
    pgto_vec: Vec<PGTO>,
    no_pgtos: usize,
    center_pos: &'a Atom,
}

#[derive(Clone, Debug)]
struct PGTO {
    alpha: f64,
    pgto_coeff: f64,
    gauss_center_pos: Array1<f64>,
    ang_mom_vec: [i32; 3],
    norm_const: f64,
}

impl<'a> BasisSet<'a> {
    pub fn new(basisset_name: &str, mol: &Molecule) -> Self {
        let basis_set_def_total = parser::BasisSetDefTotal::new(basisset_name);
        let shells: Vec<Shell<'_>> = Vec::<Shell>::new();
        for atom in mol.atoms_iter() {
            let basis_set_def = basis_set_def_total
                .get_basis_set_def_atom(atom.get_pse_sym())
                .unwrap();
            todo!();
            // TODO: Potential redesign necessarry: BasisSetDefAtom contains vectors of different lengths
            // -> better grouping reasonable? 
            
            // let prim_per_shell = basis_set_def.get_n_prim_shell();
            // let shell = Shell::new(atom, &basis_set_def_total);
            // shells.push(shell);
        }

        Self {
            name: basisset_name.to_string(),
            // no_ao,
            // no_bf,
            // shells,
            // use_pure_am,
        }
    }
}

impl<'a> Shell<'a> {
    fn new(atom: &'a Atom, basis_set_def_total: &BasisSetDefTotal) -> Self {
        let cgtos = Vec::<CGTO>::new();


        Self {
            ang_mom: 0,
            is_pure_am: false,
            cgtos: Vec::<CGTO>::new(),
        }
    }
}

impl<'a> CGTO<'a> {
    // fn new(pgto_vec: Vec<PGTO>, no_pgtos: usize) -> Self {
    //     Self { pgto_vec, no_pgtos }
    // }
}

impl PGTO {
    fn new(
        alpha: f64,
        pgto_coeff: f64,
        gauss_center_pos: &Array1<f64>,
        ang_mom_vec: &[i32; 3],
        norm_const: f64,
    ) -> Self {
        Self {
            alpha,
            pgto_coeff,
            gauss_center_pos: gauss_center_pos.to_owned(),
            ang_mom_vec: ang_mom_vec.to_owned(),
            norm_const,
        }
    }
}
