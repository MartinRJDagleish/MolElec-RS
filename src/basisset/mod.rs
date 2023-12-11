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
    is_pure_am: bool,
    cgtos: Vec<CGTO<'a>>, // == basis funcs.
    center_pos: &'a Atom,
}

#[derive(Debug)]
struct CGTO<'a> {
    pgto_vec: Vec<PGTO>,
    no_pgtos: usize,
    ang_mom_vec: [i32; 3],
    center_pos: &'a Atom,
}

#[derive(Clone, Debug)]
struct PGTO {
    alpha: f64,
    pgto_coeff: f64,
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
        let nprim_p_shell = basis_set_def_at.get_n_prim_p_shell(shell_idx);
        let mut curr_exp_coeff_idx = basis_set_def_at
            .no_prim_per_shell_iter()
            .take(shell_idx)
            .sum::<usize>();
        if curr_exp_coeff_idx > 0 {
            curr_exp_coeff_idx -= 1;
        }

        let ang_mom_triples = basis_set_def_at
            .get_ang_mom_chars(shell_idx)
            .get_ang_mom_triple();

        let no_cgtos = ang_mom_triples.len();
        let mut cgtos = Vec::<CGTO>::with_capacity(no_cgtos);

        for ang_mom_trip in ang_mom_triples {
            // let cgto = CGTO::new();
            let mut pgtos = Vec::<PGTO>::with_capacity(nprim_p_shell);
            for pgto_idx in (curr_exp_coeff_idx..curr_exp_coeff_idx + nprim_p_shell) {
                let pgto = PGTO::new(
                    basis_set_def_at.pgto_exps[pgto_idx],
                    basis_set_def_at.pgto_coeffs[pgto_idx],
                );
                pgtos.push(pgto);
            }
            let cgto = CGTO {
                pgto_vec: pgtos,
                no_pgtos: nprim_p_shell,
                ang_mom_vec: ang_mom_trip,
                center_pos: atom,
            };

            cgtos.push(cgto);
        }

        Self {
            is_pure_am: false,
            cgtos: Vec::<CGTO>::new(),
            center_pos: atom,
        }
    }
}

impl<'a> CGTO<'a> {
    pub fn new(pgto_vec: Vec<PGTO>, ang_mom_vec: [i32; 3], center_pos: &'a Atom) -> Self {
        Self {
            no_pgtos: pgto_vec.len(),
            pgto_vec,
            ang_mom_vec,
            center_pos,
        }
    }
}

impl PGTO {
    fn new(alpha: f64, pgto_coeff: f64) -> Self {
        let norm_const = 1.0; // TODO: implement norm_const
        Self {
            alpha,
            pgto_coeff,
            norm_const,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_basis_set() {
        println!("Test create_basis_set");
        let mol = Molecule::new("data/xyz/water90.xyz", 0);
        let test_basis = BasisSet::new("STO-3G", &mol); 
        println!("{:?}", test_basis);
    }
}
