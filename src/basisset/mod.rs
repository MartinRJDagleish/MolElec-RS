mod parser;

use crate::molecule::{atom::Atom, Molecule};
use ndarray::Array1;
use parser::{BasisSetDefAtom, BasisSetDefTotal};

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
    pub fn new(basisset_name: &str, mol: &'a Molecule) -> Self {
        let basis_set_def_total = parser::BasisSetDefTotal::new(basisset_name);
        // Potential speedup: Preallocate vector with correct size
        let mut shells: Vec<Shell<'_>> = Vec::<Shell>::new();
        for atom in mol.atoms_iter() {
            // TODO: Potential redesign necessarry: BasisSetDefAtom contains vectors of different lengths
            // -> better grouping reasonable?
            let basis_set_def_at = basis_set_def_total
                .get_basis_set_def_atom(atom.get_pse_sym())
                .unwrap();
            let no_shells = basis_set_def_at.get_no_shells();
            for shell_idx in 0..no_shells {
                let shell = Shell::new(atom, shell_idx, basis_set_def_at);
                shells.push(shell);
            }

            // let prim_per_shell = basis_set_def.get_n_prim_shell();
            // let shell = Shell::new(atom, &basis_set_def_total);
            // shells.push(shell);
            todo!();
        }

        // TODO: change for UHF
        Self {
            name: basisset_name.to_string(),
            no_ao: shells.len(),
            no_bf: shells.len(),
            shells,
            use_pure_am: false, // hard code for now
        }
    }
}

impl<'a> Shell<'a> {
    fn new(atom: &'a Atom, shell_idx: usize, basis_set_def_at: &BasisSetDefAtom) -> Self {
        let mut cgtos = Vec::<CGTO>::new();
        let nprim_p_shell = basis_set_def_at.get_n_prim_p_shell(shell_idx);
        let curr_exp_coeff_idx = basis_set_def_at
            .no_prim_per_shell_iter()
            .take(shell_idx)
            .sum::<usize>() - 1;
        
        // let ang_mom
        
        for pgto_idx in (curr_exp_coeff_idx..curr_exp_coeff_idx+nprim_p_shell) {
                        
        }
        
        
        // for pgto_idx in 0..prim_per_shell {
        //     let pgto_exp = basis_set_def_at.pgto_exps[pgto_idx];
        //     let pgto_coeff = basis_set_def_at.pgto_coeffs[pgto_idx];
        //     let ang_mom_char = basis_set_def_at.ang_mom_chars[pgto_idx];
        //     let ang_mom = match ang_mom_char {
        //         parser::AngMomChar::S => 0,
        //         parser::AngMomChar::P => 1,
        //         parser::AngMomChar::D => 2,
        //         parser::AngMomChar::F => 3,
        //         parser::AngMomChar::G => 4,
        //         parser::AngMomChar::H => 5,
        //         parser::AngMomChar::I => 6,
        //         parser::AngMomChar::J => 7,
        //         parser::AngMomChar::K => 8,
        //         parser::AngMomChar::L => 9,
        //         parser::AngMomChar::M => 10,
        //         parser::AngMomChar::N => 11,
        //         parser::AngMomChar::O => 12,
        //         parser::AngMomChar::SP => 0,
        //     };
        //     let ang_mom_vec = match ang_mom_char {
        //         parser::AngMomChar::S => [0, 0, 0],
        //         parser::AngMomChar::P => [1, 0, 0],
        //         parser::AngMomChar::D => [0, 1, 0],
        //         parser::AngMomChar::F => [0, 0, 1],
        //         parser::AngMomChar::G => [2, 0, 0],
        //         parser::AngMomChar::H => [0, 2, 0],
        //         parser::AngMomChar::I => [0, 0, 2],
        //         parser::AngMomChar::J => [1, 1, 0],
        //         parser::AngMomChar::K => [1, 0, 1],
        //         parser::AngMomChar::L => [0, 1, 1],
        //         parser::AngMomChar::M => [3, 0, 0],
        //         parser::AngMomChar::N => [0, 3, 0],
        //         parser::AngMomChar::O => [0, 0, 3],
        //         parser::AngMomChar::SP => [0,
        // }

        Self {
            ang_mom: 0,
            is_pure_am: false,
            cgtos: Vec::<CGTO>::new(),
            center_pos: atom,
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


#[cfg(test)]
mod tests {
    #[test]
    fn create_basis_set() {
        println!("Test create_basis_set");
    }
}


