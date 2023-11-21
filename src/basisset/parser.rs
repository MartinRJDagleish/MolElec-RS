use crate::molecule::PseElemSym;
use std::{collections::HashMap, fs::File, io::BufRead, io::BufReader};
use strum_macros::EnumString;


use nom::{
    bytes::complete::{tag, take_while},
    sequence::separated_pair,
    character::complete::multispace1,
    IResult,
};

// #[derive(PartialEq)]
#[derive(Debug, Default, EnumString)]
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

#[derive(Debug, Default)]
struct BasisSetDefAtom {
    elem_sym: PseElemSym,
    pgto_exps: Vec<f64>,
    pgto_coeffs: Vec<f64>,
    ang_mom_chars: Vec<AngMomChar>,
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
    
    fn parse_chunk_basis_set_file () {
        
    }

    fn parse_line(input: &str) -> IResult<&str, (&str, &str)> {
        separated_pair(take_while(|c: char| !c.is_whitespace()), multispace1, take_while(|c: char| !c.is_whitespace()))(input)
    }
    
    fn parse_content(input: &str) -> IResult<&str, Vec<(&str, &str)>> {
        let block_delim: &str = "****";
        let (input, _) = tag(block_delim)(input)?;
        let (input, _) = multispace1(input)?;
        let (input, parsed_content) = nom::multi::many1(Self::parse_line)(input)?;
        let (input, _) = multispace1(input)?;
        let (input, _) = tag(block_delim)(input)?;
        Ok((input, parsed_content))
    }

    pub fn parse_basis_set_file_psi4(
        basis_set_name: String,
    ) -> Result<HashMap<PseElemSym, BasisSetDefAtom>, Box<dyn std::error::Error>> {
        let basis_set_defs: HashMap<PseElemSym, BasisSetDefAtom> = HashMap::new();

        let basis_set_file_path: &str = match basis_set_name.to_ascii_lowercase().as_str() {
            "sto-3g" => "data/sto-3g.gbs",
            "sto-6g" => "data/sto-6g.gbs",
            "6-311g" => "data/6-311g.gbs",
            "6-311g*" => "data/6-311g_st.gbs",
            // "6-311g**" => "src/basis_sets/6-311g_st_st.gbs",
            // "def2-svp" => "src/basis_sets/def2-svp.gbs",
            // "def2-tzvp" => "src/basis_sets/def2-tzvp.gbs",
            // "def2-qzvp" => "src/basis_sets/def2-qzvp.gbs",
            // "cc-pvdz" => "src/basis_sets/cc-pVDZ.gbs",
            // "cc-pvtz" => "src/basis_sets/cc-pVTZ.gbs",
            _ => {
                panic!("Basis set not yet implemented!");
            }
        };

        let basis_set_file = File::open(basis_set_file_path)?;
        let reader = BufReader::new(basis_set_file);

        // skip the initial spherical / cartesian line
        for line in reader.lines().skip(1) {
            let line = line?;
            let trimmed = &line.trim();
            while trimmed.is_empty() || trimmed.starts_with('!') {
                continue;
            }
            
            // let (input, parsed_content) = Self::parse_content(trimmed)?;
            if let Ok((_, parsed_content)) = Self::parse_content(&line) {
                println!("{:?}", parsed_content);
            } 
            

        }
        // for line in reader.lines() {
        //     let line = line?;
        //     let data = line.trim();
        //     let mut line_start = 0 as char;
        //     if !data.is_empty() {
        //         line_start = data.chars().next().unwrap();
        //     }
        //     if data.starts_with('!') || data.is_empty() {
        //         continue;
        //     } else if data.starts_with(block_delimiter) {
        //         if !basis_set_def.alphas.is_empty() {
        //             //* Check if BasisSet is not empty
        //             //* Add the basis using the element symbol as key
        //             basis_set_total_def
        //                 .basis_set_defs_dict
        //                 .insert(basis_set_def.element_sym, basis_set_def);
        //         }
        //         basis_set_def = BasisSetDef::default();
        //         continue;
        //     } else if line_start.is_alphabetic() {
        //         let line_split: Vec<&str> = data.split_whitespace().collect();
        //         if line_split.len() == 2 {
        //             //* Old version with string -> new version with enum
        //             // basis_set.element_sym = line_split[0].to_string();
        //             //* New version with enum
        //             basis_set_def.element_sym =
        //                 match_pse_symb(&PSE_elem_sym_HashMap, line_split[0]);
        //             continue;
        //         } else if line_split[0] == "SP" {
        //             let no_prim1: usize = line_split[1].parse::<usize>().unwrap();
        //             basis_set_def.L_and_no_prim_tup.push((L_CHAR::SP, no_prim1));
        //         } else if line_split[0].len() > 2
        //             && (line_split[0].starts_with("l=") || line_split[0].starts_with("L="))
        //         {
        //             todo!("Add the values for L basis sets");
        //         } else {
        //             let L_letter_val = match line_split[0] {
        //                 "S" => L_CHAR::S,
        //                 "P" => L_CHAR::P,
        //                 "D" => L_CHAR::D,
        //                 "F" => L_CHAR::F,
        //                 "G" => L_CHAR::G,
        //                 "H" => L_CHAR::H,
        //                 "I" => L_CHAR::I,
        //                 "J" => L_CHAR::J,
        //                 "K" => L_CHAR::K,
        //                 "L" => L_CHAR::L,
        //                 "M" => L_CHAR::M,
        //                 "N" => L_CHAR::N,
        //                 "O" => L_CHAR::O,
        //                 _ => panic!("This letter is not supported!"),
        //             };
        //             // let L_val: usize = SPDF_HashMap.get(&L_val_char).unwrap().clone();
        //             let no_prim: usize = line_split[1].parse::<usize>().unwrap();
        //             basis_set_def
        //                 .L_and_no_prim_tup
        //                 .push((L_letter_val, no_prim));
        //         }
        //     } else {
        //         let parameters_vec = data
        //             .replace("D", "e")
        //             .split_whitespace()
        //             .map(|x| x.parse::<f64>().unwrap())
        //             .collect::<Vec<f64>>();
        //         if parameters_vec.len() > 2 {
        //             //* This is the SP basis case
        //             basis_set_def.alphas.push(parameters_vec[0]);
        //             basis_set_def.cgto_coeffs.push(parameters_vec[1]); //* Values at even positions (0,2,…) are coeffs for S, odd values are for P (1,3,…) */
        //             basis_set_def.cgto_coeffs.push(parameters_vec[2]);
        //         } else {
        //             basis_set_def.alphas.push(parameters_vec[0]);
        //             basis_set_def.cgto_coeffs.push(parameters_vec[1]);
        //         }
        //     }
        // }

        Ok(basis_set_defs)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser() {
        let basis_set_name = "sto-3g".to_string();
        let basis_set_defs = BasisSetDefTotal::new(basis_set_name);
        println!("{:?}", basis_set_defs);
    }
}