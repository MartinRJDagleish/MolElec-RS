pub mod parser;

use self::parser::AngMomChar;
use crate::molecule::{atom::Atom, Molecule};
use getset::{CopyGetters, Getters};
use parser::BasisSetDefAtom;
use std::f64::consts::PI;

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
    _6_311G_d_p,
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
#[derive(Debug, CopyGetters)]
pub(crate) struct BasisSet<'a> {
    name: String,
    use_pure_am: bool,
    #[getset(get_copy = "pub")]
    no_ao: usize,
    #[getset(get_copy = "pub")]
    no_bf: usize,
    #[getset(get_copy = "pub")]
    no_occ: usize,
    shells: Vec<Shell<'a>>,
    sh_len_offsets: Vec<usize>,
}

#[derive(Debug, CopyGetters)]
pub struct Shell<'a> {
    is_pure_am: bool,
    cgtos: Vec<CGTO<'a>>, // == basis funcs.
    ang_mom_type: AngMomChar,
    #[getset(get_copy = "pub")]
    shell_len: usize,
    center_pos: &'a Atom,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug, Getters)]
pub struct CGTO<'a> {
    pgto_vec: Vec<PGTO>,
    no_pgtos: usize,
    #[getset(get = "pub")]
    ang_mom_type: AngMomChar,
    #[getset(get = "pub")]
    ang_mom_vec: [i32; 3],
    #[getset(get = "pub")]
    centre_pos: &'a Atom,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug, CopyGetters)]
#[getset(get_copy = "pub")]
pub struct PGTO {
    alpha: f64,
    pgto_coeff: f64,
    norm_const: f64,
}

impl<'a> BasisSet<'a> {
    pub fn new(basisset_name: &str, mol: &'a Molecule) -> Self {
        let basis_set_def_total = parser::BasisSetDefTotal::new(basisset_name);
        // Potential speedup: Preallocate vector with correct size
        let mut shells: Vec<Shell<'_>> = Vec::<Shell>::new();
        let mut sh_len_offsets: Vec<usize> = vec![0]; // First element is 0
        let mut sh_offset = 0;
        let mut no_bf = 0;
        for atom in mol.atoms_iter() {
            let basis_set_def_at = basis_set_def_total
                .basis_set_def_atom(atom.pse_sym())
                .unwrap();
            let no_shells = basis_set_def_at.get_no_shells();
            for shell_idx in 0..no_shells {
                let shell = Shell::new(atom, shell_idx, basis_set_def_at);
                no_bf += shell.shell_len;
                sh_offset += shell.shell_len;
                sh_len_offsets.push(sh_offset);
                shells.push(shell);
            }
        }

        // [ ] TODO: change for UHF
        Self {
            name: basisset_name.to_string(),
            no_ao: no_bf,
            no_bf,
            no_occ: mol.no_elec() / 2,
            shells,
            sh_len_offsets,
            use_pure_am: false, // hard code for now
        }
    }

    pub fn shell_iter(&self) -> std::slice::Iter<Shell<'a>> {
        self.shells.iter()
    }

    #[inline(always)]
    pub fn shell(&self, idx: usize) -> &Shell<'a> {
        &self.shells[idx]
    }

    #[inline(always)]
    pub fn no_shells(&self) -> usize {
        self.shells.len()
    }

    #[inline(always)]
    pub fn sh_len_offset(&self, sh_idx: usize) -> usize {
        self.sh_len_offsets[sh_idx]
    }
}

impl<'a> Shell<'a> {
    fn new(atom: &'a Atom, shell_idx: usize, basis_set_def_at: &BasisSetDefAtom) -> Self {
        let curr_shell_def = &basis_set_def_at.shell_defs[shell_idx];
        let no_prim = curr_shell_def.no_prim();
        let curr_ang_mom_char = curr_shell_def.ang_mom_char();
        let ang_mom_triples = curr_ang_mom_char.get_ang_mom_triple();

        let no_cgtos = ang_mom_triples.len();
        let mut cgtos = Vec::<CGTO>::with_capacity(no_cgtos);

        let alphas = curr_shell_def.pgto_exps();
        let coeffs = curr_shell_def.pgto_coeffs();

        for ang_mom_trip in ang_mom_triples {
            let mut pgtos = Vec::<PGTO>::with_capacity(no_prim);
            for pgto_idx in 0..no_prim {
                let pgto = PGTO::new(
                    alphas[pgto_idx],
                    coeffs[pgto_idx],
                    &ang_mom_trip,
                    curr_ang_mom_char as i32,
                );
                pgtos.push(pgto);
            }
            let cgto = CGTO {
                pgto_vec: pgtos,
                no_pgtos: no_prim,
                ang_mom_type: curr_shell_def.ang_mom_char(),
                ang_mom_vec: ang_mom_trip,
                centre_pos: atom,
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
            ang_mom_type: curr_ang_mom_char,
            shell_len: no_cgtos,
            center_pos: atom,
        }
    }

    pub fn cgto_iter(&self) -> std::slice::Iter<CGTO<'a>> {
        self.cgtos.iter()
    }
}

impl<'a> CGTO<'a> {
    pub fn new(
        pgto_vec: Vec<PGTO>,
        no_pgtos: usize,
        ang_mom_type: AngMomChar,
        ang_mom_vec: [i32; 3],
        centre_pos: &'a Atom,
    ) -> Self {
        Self {
            pgto_vec,
            no_pgtos,
            ang_mom_type,
            ang_mom_vec,
            centre_pos,
        }
    }

    /// Calculate the normalization constant for a given primitive Gaussian type orbital (CGTO)
    /// add it to the norm_const field of the PGTOs
    ///
    /// - Source: Valeev -- Fundamentals of Molecular Integrals Evaluation
    /// - Link: https://arxiv.org/pdf/2007.12057.pdf
    /// - Eq. 2.25 on page 10
    #[allow(non_snake_case)]
    fn calc_cart_norm_const_cgto(&mut self) {
        lazy_static! {
            static ref PI_3_2: f64 = PI.powf(1.5);
        }
        let mut norm_const_cgto = 0.0_f64;
        let L_tot = self.ang_mom_type as u32;
        let pi_factor = *PI_3_2 / (2_i32.pow(L_tot) as f64)
            * (self.ang_mom_vec.map(|x| double_fac(2 * x - 1)))
                .iter()
                .product::<i32>() as f64;

        for pgto1 in &self.pgto_vec {
            for pgto2 in &self.pgto_vec {
                norm_const_cgto +=
                    pgto1.pgto_coeff * pgto2.pgto_coeff * pgto1.norm_const * pgto2.norm_const
                        / (pgto1.alpha + pgto2.alpha).powf(L_tot as f64 + 1.5);
            }
        }

        norm_const_cgto *= pi_factor;
        norm_const_cgto = 1.0 / (norm_const_cgto.sqrt());

        for pgto in self.pgto_vec.iter_mut() {
            pgto.norm_const *= norm_const_cgto;
        }
    }

    pub fn pgto_iter(&self) -> std::slice::Iter<PGTO> {
        self.pgto_vec.iter()
    }
}

impl PGTO {
    #[allow(non_snake_case)]
    pub fn new(alpha: f64, pgto_coeff: f64, ang_mom_vec: &[i32; 3], L_tot: i32) -> Self {
        Self {
            alpha,
            pgto_coeff,
            norm_const: Self::calc_norm_const(alpha, ang_mom_vec, L_tot),
        }
    }

    /// Calculate the normalization constant for a given primitive Gaussian type orbital (PGTO)
    ///
    /// - Source: Valeev -- Fundamentals of Molecular Integrals Evaluation
    /// - Link: https://arxiv.org/pdf/2007.12057.pdf
    /// - Using formula (2.11) on page 8
    #[allow(non_snake_case)]
    pub fn calc_norm_const(alpha: f64, ang_mom_vec: &[i32; 3], L_tot: i32) -> f64 {
        // let numerator: f64 = (2.0 * alpha / PI).powf(1.5) * (4.0 * alpha).powi(L_tot);
        // let denom: i32 = ang_mom_vec
        //     .map(|x| double_fac(2 * x - 1))
        //     .iter()
        //     .product::<i32>();
        //
        // (numerator / denom as f64).sqrt()
        // Version 2
        lazy_static! {
            static ref PI_INV_POW_3_2: f64 = 1.0 / (PI * PI.sqrt());
        }
        let numerator: f64 =
            (2.0_f64).powf((2 * L_tot) as f64 + 3.0 / 2.0) * alpha.powf(L_tot as f64 + 3.0 / 2.0);
        let denom: i32 = ang_mom_vec
            .map(|x| double_fac(2 * x - 1))
            .iter()
            .product::<i32>();
        (*PI_INV_POW_3_2 * numerator / denom as f64).sqrt()
    }
}

#[inline(always)]
fn double_fac(mut n: i32) -> i32 {
    let mut res = 1;
    match n {
        -1..=1 => 1, // (-1)!! = 0!! = 1!! = 1
        _ => {
            if n % 2 == 1 {
                while n >= 2 {
                    res *= n;
                    n -= 2;
                }
            } else {
                while n >= 1 {
                    res *= n;
                    n -= 2;
                }
            }
            res
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_basis_set() {
        let mol = Molecule::new("data/xyz/water90.xyz", 0);
        let test_basis = BasisSet::new("STO-3G", &mol);
        println!("{:?}", test_basis);
    }
    
    #[test]
    fn test_sh_len_offsets() {
        let mol = Molecule::new("data/xyz/water90.xyz", 0);
        let test_basis = BasisSet::new("STO-3G", &mol);
        
        println!("sh_len_offsets: {:?}", test_basis.sh_len_offsets);
        for shell in test_basis.shells.iter() {
            println!("Shell: {:?}\n", shell);
            println!("Type:{:?}", shell.ang_mom_type);
        }
    }
}
