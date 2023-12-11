use crate::molecule::PseElemSym;
use std::{collections::HashMap, fs::File, io::BufRead, io::BufReader, str::FromStr};
use strum_macros::EnumString;

// #[derive(PartialEq)]
#[derive(Debug, Default, EnumString, Clone)]
#[repr(i32)]
pub enum AngMomChar {
    #[default]
    S = 0,
    P = 1,
    D = 2,
    F = 3,
    G = 4,
    H = 5,
    I = 6,
    J = 7,
    SP = 10,
}

impl AngMomChar {
    pub fn get_ang_mom_triple(&self) -> Vec<[i32; 3]> {
        let max = match self {
            AngMomChar::SP => 1,
            _ => self.clone() as i32,
        };
        let mut result = Vec::with_capacity(2 * max as usize + 1);
        if let AngMomChar::SP = self {
            result.insert(0, [0, 0, 0]);
        }

        for k in 0..=max {
            for j in 0..=max {
                for i in 0..=max {
                    if i + j + k == max {
                        result.push([i, j, k]);
                    }
                }
            }
        }

        result
    }
}

#[derive(Debug, Default)]
pub struct BasisSetDefTotal {
    basis_set_name: String,
    basis_set_defs_hm: HashMap<PseElemSym, BasisSetDefAtom>,
}

#[derive(Debug, Default, Clone)]
pub struct BasisSetDefAtom {
    // elem_sym: PseElemSym, // no need to store elem_sym redundantly
    pub ang_mom_chars: Vec<AngMomChar>,
    pub no_prim_per_shell: Vec<usize>,
    pub shell_defs: Vec<BasisSetDefShell>,
}

#[derive(Debug, Default, Clone)]
pub struct BasisSetDefShell {
    ang_mom_char: AngMomChar,
    no_prim: usize,
    pgto_exps: Vec<f64>,
    pgto_coeffs: Vec<f64>,
}

impl BasisSetDefAtom {
    pub(crate) fn get_n_prim_p_shell(&self, shell_idx: usize) -> usize {
        self.no_prim_per_shell[shell_idx]
    }

    pub(crate) fn get_no_shells(&self) -> usize {
        self.no_prim_per_shell.len()
    }

    pub(crate) fn no_prim_per_shell_iter(&self) -> std::slice::Iter<usize> {
        self.no_prim_per_shell.iter()
    }
}

impl BasisSetDefShell {
    fn split_sp_shell(shell: &Self) -> (Self, Self) {
        let mut shell_s = Self::default();
        let mut shell_p = Self::default();
        shell_s.ang_mom_char = AngMomChar::S;
        shell_p.ang_mom_char = AngMomChar::P;
        shell_s.no_prim = shell.no_prim;
        shell_p.no_prim = shell.no_prim;
        shell_s.pgto_exps = shell.pgto_exps.clone();
        shell_p.pgto_exps = shell.pgto_exps.clone();

        for idx in 0..shell.pgto_coeffs.len() / 2 {
            // even for s
            shell_s.pgto_coeffs.push(shell.pgto_coeffs[2 * idx]);
            // odd for p
            shell_p.pgto_coeffs.push(shell.pgto_coeffs[2 * idx + 1]);
        }

        (shell_s, shell_p)
    }

    pub fn get_pgtos_exps(&self) -> &[f64] {
        &self.pgto_exps
    }

    pub fn get_pgto_coeffs(&self) -> &[f64] {
        &self.pgto_coeffs
    }
    pub fn get_no_prim(&self) -> usize {
        self.no_prim
    }
    pub fn get_ang_mom_char(&self) -> &AngMomChar {
        &self.ang_mom_char
    }
}

impl BasisSetDefTotal {
    pub fn new(basis_set_name: &str) -> Self {
        let basis_set_defs: HashMap<PseElemSym, BasisSetDefAtom> =
            Self::parse_basis_set_file_psi4(basis_set_name).unwrap();

        Self {
            basis_set_name: basis_set_name.to_string(),
            basis_set_defs_hm: basis_set_defs,
        }
    }

    fn parse_basis_set_file_psi4(
        basis_set_name: &str,
    ) -> Result<HashMap<PseElemSym, BasisSetDefAtom>, Box<dyn std::error::Error>> {
        let mut basis_set_defs: HashMap<PseElemSym, BasisSetDefAtom> = HashMap::new();

        let basis_set_file_path: &str = match basis_set_name.to_ascii_lowercase().as_str() {
            "sto-3g" => "data/basis/sto-3g.gbs",
            "sto-6g" => "data/basis/sto-6g.gbs",
            "6-311g" => "data/basis/6-311g.gbs",
            "6-311g*" => "data/basis/6-311g_st.gbs",
            "6-311g**" => "data/basis/6-311g_st_st.gbs",
            "def2-svp" => "data/basis/def2-svp.gbs",
            "def2-tzvp" => "data/basis/def2-tzvp.gbs",
            "def2-qzvp" => "data/basis/def2-qzvp.gbs",
            "cc-pvdz" => "data/basis/cc-pvdz.gbs",
            "cc-pvtz" => "data/basis/cc-pvtz.gbs",
            _ => {
                panic!("Basis set not yet implemented!");
            }
        };

        let block_delimiter: &str = "****";

        let basis_set_file = File::open(basis_set_file_path)?;
        let reader = BufReader::new(basis_set_file);

        let mut basis_set_def_atom: BasisSetDefAtom = BasisSetDefAtom::default();
        let mut basis_set_def_shell: BasisSetDefShell = BasisSetDefShell::default();
        let mut elem_sym = PseElemSym::default();

        for line in reader.lines().skip(1) {
            let line = line?;
            let data_line = line.trim();
            //1. Skip initial empty lines and comments
            if data_line.is_empty() || data_line.starts_with('!') {
                continue;
            }
            //2. Check if the line starts with the block delimiter
            else if data_line.starts_with(block_delimiter) {
                // Check if previous basis_set_def_atom is done
                if !basis_set_def_shell.pgto_coeffs.is_empty() {
                    if let AngMomChar::SP = basis_set_def_shell.ang_mom_char {
                        let (s_shell, p_shell) =
                            BasisSetDefShell::split_sp_shell(&basis_set_def_shell);
                        basis_set_def_atom.shell_defs.push(s_shell);
                        basis_set_def_atom.shell_defs.push(p_shell);
                    } else {
                        basis_set_def_atom
                            .shell_defs
                            .push(basis_set_def_shell.clone());
                    }
                    // Add the basis using the element symbol as key
                    basis_set_defs.insert(elem_sym, basis_set_def_atom);
                    // Create new basis_set_def_atom
                    basis_set_def_atom = BasisSetDefAtom::default();
                    basis_set_def_shell = BasisSetDefShell::default();
                }
            } else if data_line.chars().next().unwrap().is_alphabetic() {
                //3. Check if the line starts with an PseElemSymb or AngMomChar
                let line_split: Vec<&str> = data_line.split_whitespace().collect();
                if line_split.len() == 2 {
                    //* New version with enum
                    elem_sym = match PseElemSym::from_str(line_split[0]) {
                        Ok(elem_sym) => elem_sym,
                        Err(e) => panic!("Error: {}", e),
                    };
                    continue;
                } else {
                    // Save the previous shell
                    if !basis_set_def_shell.pgto_coeffs.is_empty() {
                        if let AngMomChar::SP = basis_set_def_shell.ang_mom_char {
                            let (s_shell, p_shell) =
                                BasisSetDefShell::split_sp_shell(&basis_set_def_shell);
                            basis_set_def_atom.shell_defs.push(s_shell);
                            basis_set_def_atom.shell_defs.push(p_shell);
                            basis_set_def_shell = BasisSetDefShell::default();
                        } else {
                            basis_set_def_atom.shell_defs.push(basis_set_def_shell);
                            basis_set_def_shell = BasisSetDefShell::default();
                        }
                    }
                    let ang_mom_char = AngMomChar::from_str(line_split[0])?;
                    let no_prim: usize = line_split[1].parse::<usize>()?;
                    basis_set_def_shell.ang_mom_char = ang_mom_char.clone();
                    basis_set_def_shell.no_prim = no_prim;
                    if let AngMomChar::SP = ang_mom_char {
                        basis_set_def_atom.ang_mom_chars.push(AngMomChar::S);
                        basis_set_def_atom.ang_mom_chars.push(AngMomChar::P);
                        basis_set_def_atom.no_prim_per_shell.push(no_prim);
                        basis_set_def_atom.no_prim_per_shell.push(no_prim);
                    } else {
                        basis_set_def_atom.ang_mom_chars.push(ang_mom_char);
                        basis_set_def_atom.no_prim_per_shell.push(no_prim);
                    }
                }
            } else {
                let parameters_vec = data_line
                    .replace('D', "e")
                    .split_whitespace()
                    .map(|x| x.parse::<f64>().unwrap())
                    .collect::<Vec<f64>>();
                if parameters_vec.len() > 2 {
                    //* This is the SP basis case
                    basis_set_def_shell.pgto_exps.push(parameters_vec[0]);
                    basis_set_def_shell.pgto_coeffs.push(parameters_vec[1]); //* Values at even positions (0,2,…) are coeffs for S, odd values are for P (1,3,…) */
                    basis_set_def_shell.pgto_coeffs.push(parameters_vec[2]);
                } else {
                    basis_set_def_shell.pgto_exps.push(parameters_vec[0]);
                    basis_set_def_shell.pgto_coeffs.push(parameters_vec[1]);
                }
            }
        }

        Ok(basis_set_defs)
    }

    pub fn get_basis_set_def_atom(&self, elem_sym: &PseElemSym) -> Option<&BasisSetDefAtom> {
        self.basis_set_defs_hm.get(elem_sym)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser1() {
        let basis_set_name = "cc-pVTZ";
        let basis_set_defs = BasisSetDefTotal::new(basis_set_name);
        println!("{:?}", basis_set_defs.basis_set_defs_hm.get(&PseElemSym::H));
        // println!("{:?}", basis_set_defs);
    }

    #[test]
    fn test_parser2() {
        let basis_set_name = "sto-3g";
        let basis_set_defs = BasisSetDefTotal::new(basis_set_name);
        println!("{:?}", basis_set_defs.basis_set_defs_hm.get(&PseElemSym::H));
        // println!("{:?}", basis_set_defs);
    }

    #[test]
    fn test_parser3() {
        let basis_set_name = "sto-3g";
        let basis_set_defs = BasisSetDefTotal::new(basis_set_name);
        // println!("\n{:?}\n", basis_set_defs);
        let basis_set_def_at = basis_set_defs
            .basis_set_defs_hm
            .get(&PseElemSym::O)
            .unwrap();
        println!("{:?}", basis_set_def_at);
        assert!(basis_set_def_at.ang_mom_chars.len() == 3);
        assert!(basis_set_def_at.no_prim_per_shell.len() == 3);
    }

    #[test]
    fn test_ang_mom_char() {
        let ang_mom_char = AngMomChar::from_str("D").unwrap();
        println!("{:?}", ang_mom_char.get_ang_mom_triple());
    }
}
