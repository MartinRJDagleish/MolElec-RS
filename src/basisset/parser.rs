use crate::molecule::PseElemSym;
use getset::{CopyGetters, Getters};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::BufRead,
    io::BufReader,
    path::PathBuf,
    str::FromStr,
};
use strum_macros::EnumString;

#[derive(Debug, Default, EnumString, Clone, Copy)]
#[repr(i32)]
pub(crate) enum AngMomChar {
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
            _ => *self as i32,
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

#[derive(Debug, Default, Clone, Getters, CopyGetters)]
pub struct BasisSetDefShell {
    #[getset(get_copy = "pub")]
    ang_mom_char: AngMomChar,
    #[getset(get_copy = "pub")]
    no_prim: usize,
    #[getset(get = "pub")]
    pgto_exps: Vec<f64>,
    #[getset(get = "pub")]
    pgto_coeffs: Vec<f64>,
}

impl BasisSetDefAtom {
    pub(crate) fn get_no_shells(&self) -> usize {
        self.no_prim_per_shell.len()
    }
}

impl BasisSetDefShell {
    fn split_sp_shell(inp_shell: &Self) -> (Self, Self) {
        let mut shelldef_s = Self::default();
        let mut shelldef_p = Self::default();
        shelldef_s.ang_mom_char = AngMomChar::S;
        shelldef_p.ang_mom_char = AngMomChar::P;
        shelldef_s.no_prim = inp_shell.no_prim;
        shelldef_p.no_prim = inp_shell.no_prim;
        shelldef_s.pgto_exps = inp_shell.pgto_exps.clone();
        shelldef_p.pgto_exps = inp_shell.pgto_exps.clone();

        for idx in 0..inp_shell.pgto_coeffs.len() / 2 {
            // even for S
            shelldef_s.pgto_coeffs.push(inp_shell.pgto_coeffs[2 * idx]);
            // odd for P
            shelldef_p
                .pgto_coeffs
                .push(inp_shell.pgto_coeffs[2 * idx + 1]);
        }

        (shelldef_s, shelldef_p)
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

    fn find_basis_set_file_path(basis_set_name: &str) -> Result<PathBuf, &'static str> {
        let target_file_name = format!(
            "{}.gbs",
            basis_set_name.replace('*', "_st").to_ascii_lowercase()
        );

        let entries =
            fs::read_dir("data/basis").map_err(|_| "Could not read the basis set directory")?;

        for entry in entries {
            let entry = entry.map_err(|_| "Error reading directory entry")?;
            let path = entry.path();
            if let Some(file_name) = path.file_name().and_then(|s| s.to_str()) {
                if file_name.to_ascii_lowercase() == target_file_name {
                    return Ok(path);
                }
            }
        }

        Err("Basis set not yet implemented")
    }
    fn parse_basis_set_file_psi4(
        basis_set_name: &str,
    ) -> Result<HashMap<PseElemSym, BasisSetDefAtom>, Box<dyn std::error::Error>> {
        let mut basis_set_defs: HashMap<PseElemSym, BasisSetDefAtom> = HashMap::new();

        let basis_set_file_path = Self::find_basis_set_file_path(basis_set_name)?;
        const BLOCK_DELIM: &str = "****"; // for .gbs files -> Psi4 format

        let basis_set_file = File::open(basis_set_file_path)?;
        let reader = BufReader::new(basis_set_file);

        let mut basis_set_def_atom = BasisSetDefAtom::default();
        let mut basis_set_def_shell = BasisSetDefShell::default();
        let mut elem_sym = PseElemSym::default();

        for line in reader.lines().skip(1) {
            let line = line?;
            let data_line = line.trim();
            //1. Skip initial empty lines and comments
            if data_line.is_empty() || data_line.starts_with('!') {
                continue;
            }
            //2. Check if the line starts with the block delimiter
            else if data_line.starts_with(BLOCK_DELIM) {
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
                    basis_set_def_shell.ang_mom_char = ang_mom_char;
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

    pub fn basis_set_def_atom(&self, elem_sym: &PseElemSym) -> Option<&BasisSetDefAtom> {
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
