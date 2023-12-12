mod parser;

use std::f64::consts::PI;

use crate::molecule::{atom::Atom, Molecule};
use parser::BasisSetDefAtom;


#[allow(non_camel_case_types)]
#[derive(Debug)]
enum BasisSetVariants {
    STO_3G, 
    STO_6G,
    _6_31G,
    _6_311G,
    cc_pVDZ,
    cc_pVTZ,
    def2_SVP,
    def2_TZVP,
    _6_311_plus_plus_G,
    _6_311_plus_plus_G_star,
    _6_311_plus_G_star,
    _6_311G_d_p
}

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
pub struct Shell<'a> {
    is_pure_am: bool,
    cgtos: Vec<CGTO<'a>>, // == basis funcs.
    center_pos: &'a Atom,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
pub struct CGTO<'a> {
    pgto_vec: Vec<PGTO>,
    no_pgtos: usize,
    ang_mom_vec: [i32; 3],
    center_pos: &'a Atom,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug)]
pub struct PGTO {
    alpha: f64,
    pgto_coeff: f64,
    norm_const: f64,
}

impl PGTO {
    #[inline(always)]
    pub(crate) fn alpha(&self) -> f64 {
        self.alpha
    }

    #[inline(always)]
    pub(crate) fn pgto_coeff(&self) -> f64 {
        self.pgto_coeff
    }

    #[inline(always)]
    pub(crate) fn norm_const(&self) -> f64 {
        self.norm_const
    }

}

impl<'a> BasisSet<'a> {
    pub fn new(basisset_name: &str, mol: &'a Molecule) -> Self {
        let basis_set_def_total = parser::BasisSetDefTotal::new(basisset_name);
        // Potential speedup: Preallocate vector with correct size
        let mut shells: Vec<Shell<'_>> = Vec::<Shell>::new();
        for atom in mol.atoms_iter() {
            let basis_set_def_at = basis_set_def_total
                .basis_set_def_atom(atom.pse_sym())
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

    pub fn shell_iter(&self) -> std::slice::Iter<Shell<'a>> {
        self.shells.iter()
    }
}

impl<'a> Shell<'a> {
    fn new(atom: &'a Atom, shell_idx: usize, basis_set_def_at: &BasisSetDefAtom) -> Self {
        let curr_shell_def = &basis_set_def_at.shell_defs[shell_idx];
        let no_prim = curr_shell_def.no_prim();
        let ang_mom_triples = curr_shell_def.ang_mom_char().get_ang_mom_triple();

        let no_cgtos = ang_mom_triples.len();
        let mut cgtos = Vec::<CGTO>::with_capacity(no_cgtos);

        let alphas = curr_shell_def.pgto_exps();
        let coeffs = curr_shell_def.pgto_coeffs();

        for ang_mom_trip in ang_mom_triples {
            let mut pgtos = Vec::<PGTO>::with_capacity(no_prim);
            for pgto_idx in 0..no_prim {
                let pgto = PGTO::new(alphas[pgto_idx], coeffs[pgto_idx], &ang_mom_trip);
                pgtos.push(pgto);
            }
            let cgto = CGTO {
                pgto_vec: pgtos,
                no_pgtos: no_prim,
                ang_mom_vec: ang_mom_trip,
                center_pos: atom,
            };

            cgtos.push(cgto);
        }

        // Calc norm const for CGTOs
        for cgto in cgtos.iter_mut() {
            cgto.calc_cart_norm_const_cgto();
        }

        Self {
            is_pure_am: false,
            cgtos,
            center_pos: atom,
        }
    }
    
    pub fn cgto_iter(&self) -> std::slice::Iter<CGTO<'a>> {
        self.cgtos.iter()
    }
}

impl<'a> CGTO<'a> {
    pub fn new(mut pgto_vec: Vec<PGTO>, ang_mom_vec: [i32; 3], center_pos: &'a Atom) -> Self {
        Self {
            no_pgtos: pgto_vec.len(),
            pgto_vec,
            ang_mom_vec,
            center_pos,
        }
    }

    /// Calculate the normalization constant for a given primitive Gaussian type orbital (CGTO)
    /// add it to the norm_const field of the PGTOs
    ///
    /// Source: Valeev -- Fundamentals of Molecular Integrals Evaluation
    /// Link: https://arxiv.org/pdf/2007.12057.pdf
    /// Eq. 2.25 on page 10
    fn calc_cart_norm_const_cgto(&mut self) {
        let mut norm_const_cgto = 0.0_f64;

        let L_sum = self.ang_mom_vec.iter().sum::<i32>();
        let pi_factor = PI.powf(1.5) / (2.0_f64.powi(L_sum))
            * (self.ang_mom_vec.map(|x| double_fac(2 * x - 1)))
                .iter()
                .product::<i32>() as f64;

        for pgto1 in &self.pgto_vec {
            for pgto2 in &self.pgto_vec {
                norm_const_cgto +=
                    pgto1.pgto_coeff * pgto2.pgto_coeff * pgto1.norm_const * pgto2.norm_const
                        / (pgto1.alpha + pgto2.alpha).powf(L_sum as f64 + 1.5);
            }
        }

        norm_const_cgto *= pi_factor;
        norm_const_cgto = norm_const_cgto.powf(-0.5);

        for pgto in self.pgto_vec.iter_mut() {
            pgto.norm_const *= norm_const_cgto;
        }
    }
    
    pub fn pgto_iter(&self) -> std::slice::Iter<PGTO> {
        self.pgto_vec.iter()
    }
    
    pub fn centre_pos(&self) -> &'a Atom{
        &self.center_pos
    }
    
    pub fn ang_mom_vec(&self) -> &[i32; 3] {
        &self.ang_mom_vec
    }
}

impl PGTO {
    pub fn new(alpha: f64, pgto_coeff: f64, ang_mom_vec: &[i32; 3]) -> Self {
        Self {
            alpha,
            pgto_coeff,
            norm_const: Self::calc_norm_const(alpha, ang_mom_vec),
        }
    }

    /// Calculate the normalization constant for a given primitive Gaussian type orbital (PGTO)
    ///
    /// Source: Valeev -- Fundamentals of Molecular Integrals Evaluation
    /// Link: https://arxiv.org/pdf/2007.12057.pdf
    /// Using formula (2.11) on page 8
    pub fn calc_norm_const(alpha: f64, ang_mom_vec: &[i32; 3]) -> f64 {
        let l_sum = ang_mom_vec.iter().sum::<i32>();
        let numerator: f64 = (2.0 * alpha / PI).powf(1.5) * (4.0 * alpha).powi(l_sum);
        let denom: i32 = ang_mom_vec
            .map(|x| double_fac(2 * x - 1))
            .iter()
            .product::<i32>();

        (numerator / denom as f64).sqrt()
    }
}

#[inline(always)]
fn double_fac(n: i32) -> i32 {
    match n {
        -1 => 1,
        0 => 1,
        1 => 1,
        _ => n * double_fac(n - 2),
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
