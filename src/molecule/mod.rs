use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use atom::Atom;
use ndarray::prelude::*;

mod atom;

const PSE_ELEM_SYMBS: [&str; 119] = [
    "Du", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];

#[derive(Debug,Default)]
pub struct Molecule {
    tot_charge: i32,
    atoms: Vec<Atom>,
    geom: Geometry,
    z_vals: Vec<u32>,
    no_elec: u32,
    no_atoms: usize,
}

#[derive(Debug, Default)]
struct Geometry {
    coords_matr: Array2<f64>,
}

impl Geometry {
    fn new(geom_matr: Array2<f64>) -> Self {
        Self {
            coords_matr: geom_matr,
        }
    }
}

#[allow(non_snake_case)]
impl Molecule {
    pub fn new(geom_filepath: &str, charge: i32) -> Self {
        let tot_charge = charge;
        let atoms = Vec::new();
        let (z_vals, geom_matr, no_atoms) = Self::read_xyz_xmol_inputfile(geom_filepath).unwrap();
        let geom = Geometry::new(geom_matr);
        let no_elec = z_vals.iter().sum::<u32>() + tot_charge as u32;

        Self {
            tot_charge,
            atoms,
            geom,
            z_vals,
            no_elec,
            no_atoms,
        }
    }

    pub fn read_xyz_xmol_inputfile(
        geom_filename: &str,
    ) -> Result<(Vec<u32>, Array2<f64>, usize), Box<dyn std::error::Error>> {
        println!("Inputfile: {geom_filename}");
        println!("Reading geometry from input file...\n");

        let geom_file = File::open(geom_filename)?;
        let reader = BufReader::new(geom_file);
        let mut lines = reader
            .lines()
            .map(|line| line.expect("Failed to read line!"));

        // let mut first_line = String::new();
        // reader.read_line(&mut first_line);
        
        let no_atoms: usize = lines.next().unwrap().trim().parse()?;

        // let mut z_vals: Vec<u32> = Vec::new();
        let mut at_symbs: Vec<String> = Vec::new();
        let mut geom_matr: Array2<f64> = Array2::zeros((no_atoms, 3));

        // let mut line: String;
        // skip the comment line
        for (at_idx, line) in lines.skip(1).enumerate() {
            let mut line_parts = line.trim().split_whitespace();

            // at_symbs.push(line_parts.next().unwrap().to_string());
            at_symbs.push(line_parts.next().unwrap().to_string());
            for cc in 0..3 {
                // geom_matr[(at_idx, cc)] = line_parts[cc + 1].parse()?;
                geom_matr[(at_idx, cc)] = line_parts.next().unwrap().parse().unwrap();
            }
        }

        //* Create z_vals from atom symbols above */
        let mut z_vals = Vec::new();
        for atom in at_symbs {
            let z_val = PSE_ELEM_SYMBS
                .iter()
                .position(|&sy| sy == atom)
                .unwrap_or(0);
            z_vals.push(z_val as u32);
        }

        //* Convert geom_matr from Angstrom to Bohr (atomic units) */
        const AA_TO_BOHR: f64 = 1.0e-10 / physical_constants::BOHR_RADIUS;
        geom_matr.mapv_inplace(|x| x * AA_TO_BOHR);

        Ok((z_vals, geom_matr, no_atoms))
    }

    #[inline(always)]
    fn no_atoms(self) -> usize {
        self.atoms.len()
    }
    
    // #[inline]
    // fn 

    // fn atom(self, idx: usize) -> &Atom {
    //     &self.atoms[idx]
    // }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mol_create() {
        let water_90_fpath = "data/xyz/water90.xyz";
        let test_mol = Molecule::new(water_90_fpath, 0);
        // assert_eq!(test_mol)
        // print_hello(input);
    }
}