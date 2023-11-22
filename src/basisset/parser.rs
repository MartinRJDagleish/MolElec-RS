use crate::molecule::PseElemSym;
use std::{collections::HashMap, fs::File, io::BufRead, io::BufReader, str::FromStr};
use strum_macros::EnumString;

// #[derive(PartialEq)]
#[derive(Debug, Default, EnumString, Clone)]
enum AngMomChar {
    #[default]
    S,
    P,
    D,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    SP,
}

#[derive(Debug, Default)]
pub struct BasisSetDefTotal {
    basis_set_name: String,
    basis_set_defs: HashMap<PseElemSym, BasisSetDefAtom>,
}

#[derive(Debug, Default, Clone)]
struct BasisSetDefAtom {
    elem_sym: PseElemSym,
    pgto_exps: Vec<f64>,
    pgto_coeffs: Vec<f64>,
    ang_mom_chars: Vec<AngMomChar>,
    no_prim_per_shell: Vec<usize>,
}

impl BasisSetDefTotal {
    pub fn new(basis_set_name: String) -> Self {
        let basis_set_defs: HashMap<PseElemSym, BasisSetDefAtom> =
            Self::parse_basis_set_file_psi4(basis_set_name.clone()).unwrap();

        Self {
            basis_set_name,
            basis_set_defs,
        }
    }

    pub fn add_basis_set_def_atom(&mut self, basis_set_def_atom: BasisSetDefAtom) {
        self.basis_set_defs
            .insert(basis_set_def_atom.elem_sym, basis_set_def_atom);
    }

    pub fn parse_basis_set_file_psi4(
        basis_set_name: String,
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
                if !basis_set_def_atom.pgto_coeffs.is_empty() {
                    // Add the basis using the element symbol as key
                    basis_set_defs.insert(
                        basis_set_def_atom.elem_sym.clone(),
                        basis_set_def_atom.clone(),
                    );
                } else {
                    // Create new basis_set_def_atom
                    basis_set_def_atom = BasisSetDefAtom::default();
                }
            } else if data_line.chars().next().unwrap().is_alphabetic() {
                //3. Check if the line starts with an PseElemSymb or AngMomChar
                let line_split: Vec<&str> = data_line.split_whitespace().collect();
                if line_split.len() == 2 {
                    //* New version with enum
                    basis_set_def_atom.elem_sym = match PseElemSym::from_str(line_split[0]) {
                        Ok(elem_sym) => elem_sym,
                        Err(e) => panic!("Error: {}", e),
                    };
                    continue;
                } else {
                    let ang_mom_char = AngMomChar::from_str(line_split[0])?;
                    let no_prim: usize = line_split[1].parse::<usize>()?;
                    basis_set_def_atom.ang_mom_chars.push(ang_mom_char);
                    basis_set_def_atom.no_prim_per_shell.push(no_prim);
                }
            } else {
                let parameters_vec = data_line
                    .replace("D", "e")
                    .split_whitespace()
                    .map(|x| x.parse::<f64>().unwrap())
                    .collect::<Vec<f64>>();
                if parameters_vec.len() > 2 {
                    //* This is the SP basis case
                    basis_set_def_atom.pgto_exps.push(parameters_vec[0]);
                    basis_set_def_atom.pgto_coeffs.push(parameters_vec[1]); //* Values at even positions (0,2,…) are coeffs for S, odd values are for P (1,3,…) */
                    basis_set_def_atom.pgto_coeffs.push(parameters_vec[2]);
                } else {
                    basis_set_def_atom.pgto_exps.push(parameters_vec[0]);
                    basis_set_def_atom.pgto_coeffs.push(parameters_vec[1]);
                }
            }
        }

        Ok(basis_set_defs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser() {
        let basis_set_name = "cc-pVTZ".to_string();
        let basis_set_defs = BasisSetDefTotal::new(basis_set_name);
        println!("{:?}", basis_set_defs);
    }
}
