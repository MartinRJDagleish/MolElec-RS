mod atom;

use atom::Atom;
use ndarray::linalg::Dot;
use ndarray::{s, Array1, Array2, Axis};
use ndarray_linalg::{Eigh, InverseH, UPLO};
use std::collections::HashMap;
use std::str::FromStr;
use std::{
    fs::File,
    io::{BufRead, BufReader},
};
use strum_macros::EnumString;

pub(crate) enum Cartesian {
    X = 0,
    Y = 1,
    Z = 2,
}

pub(crate) const PSE_ELEM_SYMS_STR: [&str; 119] = [
    "Du", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];

#[rustfmt::skip]
#[derive(Debug, Default, Hash, Eq, PartialEq, Clone, Copy, EnumString)]
pub(crate) enum PseElemSym {
    #[strum(serialize = "Du", serialize = "DUMMY")]
    #[default]
    Du, // Dummy atom
    H,                                                                 He, 
    Li, Be,                                         B,  C, N,  O,   F, Ne, 
    Na, Mg,                                        Al, Si, P,  S,  Cl, Ar, 
    K,  Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge, As, Se, Br, Kr, 
    Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, In, Sn, Sb, Te,  I, Xe, 
    Cs, Ba, 
        La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu,
            Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg,     Tl, Pb, Bi, Po, At, Rn, 
    Fr, Ra, 
        Ac, Th, Pa, U, Np, Pu, Am, Cm, Bk, Cf, Es, Fm, Md, No, Lr, 
               Rf, Db, Sg, Bh, Hs, Mt, Ds, Rg, Cn, Nh, Fl, Mc, Lv, Ts, Og,
}

lazy_static! {
    static ref PSE_ELEM_Z_VAL_HMAP: HashMap<PseElemSym, u32> = HashMap::from([
        (PseElemSym::H, 1),
        (PseElemSym::He, 2),
        (PseElemSym::Li, 3),
        (PseElemSym::Be, 4),
        (PseElemSym::B, 5),
        (PseElemSym::C, 6),
        (PseElemSym::N, 7),
        (PseElemSym::O, 8),
        (PseElemSym::F, 9),
        (PseElemSym::Ne, 10),
        (PseElemSym::Na, 11),
        (PseElemSym::Mg, 12),
        (PseElemSym::Al, 13),
        (PseElemSym::Si, 14),
        (PseElemSym::P, 15),
        (PseElemSym::S, 16),
        (PseElemSym::Cl, 17),
        (PseElemSym::Ar, 18),
        (PseElemSym::K, 19),
        (PseElemSym::Ca, 20),
        (PseElemSym::Sc, 21),
        (PseElemSym::Ti, 22),
        (PseElemSym::V, 23),
        (PseElemSym::Cr, 24),
        (PseElemSym::Mn, 25),
        (PseElemSym::Fe, 26),
        (PseElemSym::Co, 27),
        (PseElemSym::Ni, 28),
        (PseElemSym::Cu, 29),
        (PseElemSym::Zn, 30),
        (PseElemSym::Ga, 31),
        (PseElemSym::Ge, 32),
        (PseElemSym::As, 33),
        (PseElemSym::Se, 34),
        (PseElemSym::Br, 35),
        (PseElemSym::Kr, 36),
        (PseElemSym::Rb, 37),
        (PseElemSym::Sr, 38),
        (PseElemSym::Y, 39),
        (PseElemSym::Zr, 40),
        (PseElemSym::Nb, 41),
        (PseElemSym::Mo, 42),
        (PseElemSym::Tc, 43),
        (PseElemSym::Ru, 44),
        (PseElemSym::Rh, 45),
        (PseElemSym::Pd, 46),
        (PseElemSym::Ag, 47),
        (PseElemSym::Cd, 48),
        (PseElemSym::In, 49),
        (PseElemSym::Sn, 50),
        (PseElemSym::Sb, 51),
        (PseElemSym::Te, 52),
        (PseElemSym::I, 53),
        (PseElemSym::Xe, 54),
        (PseElemSym::Cs, 55),
        (PseElemSym::Ba, 56),
        (PseElemSym::La, 57),
        (PseElemSym::Ce, 58),
        (PseElemSym::Pr, 59),
        (PseElemSym::Nd, 60),
        (PseElemSym::Pm, 61),
        (PseElemSym::Sm, 62),
        (PseElemSym::Eu, 63),
        (PseElemSym::Gd, 64),
        (PseElemSym::Tb, 65),
        (PseElemSym::Dy, 66),
        (PseElemSym::Ho, 67),
        (PseElemSym::Er, 68),
        (PseElemSym::Tm, 69),
        (PseElemSym::Yb, 70),
        (PseElemSym::Lu, 71),
        (PseElemSym::Hf, 72),
        (PseElemSym::Ta, 73),
        (PseElemSym::W, 74),
        (PseElemSym::Re, 75),
        (PseElemSym::Os, 76),
        (PseElemSym::Ir, 77),
        (PseElemSym::Pt, 78),
        (PseElemSym::Au, 79),
        (PseElemSym::Hg, 80),
        (PseElemSym::Tl, 81),
        (PseElemSym::Pb, 82),
        (PseElemSym::Bi, 83),
        (PseElemSym::Po, 84),
        (PseElemSym::At, 85),
        (PseElemSym::Rn, 86),
        (PseElemSym::Fr, 87),
        (PseElemSym::Ra, 88),
        (PseElemSym::Ac, 89),
        (PseElemSym::Th, 90),
        (PseElemSym::Pa, 91),
        (PseElemSym::U, 92),
        (PseElemSym::Np, 93),
        (PseElemSym::Pu, 94),
        (PseElemSym::Am, 95),
        (PseElemSym::Cm, 96),
        (PseElemSym::Bk, 97),
        (PseElemSym::Cf, 98),
        (PseElemSym::Es, 99),
        (PseElemSym::Fm, 100),
        (PseElemSym::Md, 101),
        (PseElemSym::No, 102),
        (PseElemSym::Lr, 103),
        (PseElemSym::Rf, 104),
        (PseElemSym::Db, 105),
        (PseElemSym::Sg, 106),
        (PseElemSym::Bh, 107),
        (PseElemSym::Hs, 108),
        (PseElemSym::Mt, 109),
        (PseElemSym::Ds, 110),
        (PseElemSym::Rg, 111),
        (PseElemSym::Cn, 112),
        (PseElemSym::Nh, 113),
        (PseElemSym::Fl, 114),
        (PseElemSym::Mc, 115),
        (PseElemSym::Lv, 116),
        (PseElemSym::Ts, 117),
        (PseElemSym::Og, 118)
    ]);
}

#[derive(Debug, Default)]
pub struct Molecule {
    tot_charge: i32,
    tot_mass: f64,
    atoms: Vec<Atom>,
    geom: Geometry,
    z_vals: Vec<u32>,
    no_elec: usize,
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
        let (z_vals, geom_matr, no_atoms) = Self::read_xyz_xmol_inputfile(geom_filepath).unwrap();
        let no_elec = z_vals.iter().sum::<u32>() as usize + tot_charge as usize;

        //* Create atoms from input
        let mut atoms: Vec<Atom> = Vec::with_capacity(no_atoms);
        Self::create_atoms_from_input(&mut atoms, &z_vals, &geom_matr);

        let geom = Geometry::new(geom_matr);

        //* Create total mass
        let tot_mass = atoms.iter().map(|at| at.mass).sum::<f64>();

        Self {
            tot_charge,
            tot_mass,
            atoms,
            geom,
            z_vals,
            no_elec,
            no_atoms,
        }
    }

    fn create_atoms_from_input(atoms: &mut Vec<Atom>, z_vals: &Vec<u32>, geom_matr: &Array2<f64>) {
        for (at_idx, z_val) in z_vals.iter().enumerate() {
            let atom = Atom::new(
                geom_matr[(at_idx, 0)],
                geom_matr[(at_idx, 1)],
                geom_matr[(at_idx, 2)],
                *z_val,
            );
            atoms.push(atom);
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

        let no_atoms: usize = lines.next().unwrap().trim().parse()?;

        let mut at_symbs: Vec<String> = Vec::with_capacity(no_atoms);
        let mut geom_matr: Array2<f64> = Array2::zeros((no_atoms, 3));

        // skip the comment line
        for (at_idx, line) in lines.skip(1).enumerate() {
            let mut line_parts = line.split_whitespace(); // split whitespace does "trim" automatically

            at_symbs.push(line_parts.next().unwrap().to_string());
            for cc in 0..3 {
                geom_matr[(at_idx, cc)] = line_parts.next().unwrap().parse().unwrap();
            }
        }

        //* Create z_vals from atom symbols above */
        let mut z_vals: Vec<u32> = Vec::with_capacity(no_atoms);
        for atom in at_symbs {
            // Version 1 (search through array O(N) algorithm)
            // let z_val = PSE_ELEM_SYMS_STR
            //     .iter()
            //     .position(|&sy| sy == atom)
            //     .unwrap_or(0);

            // Version 2 (search through static HashMap O(1) algorithm)
            let pse_sym = PseElemSym::from_str(&atom)
                .expect("PseElemSym does not exist; check your input again!");
            let z_val = PSE_ELEM_Z_VAL_HMAP.get(&pse_sym).unwrap_or(&0).to_owned();
            z_vals.push(z_val);
        }

        //* Convert geom_matr from Angstrom to Bohr (atomic units) */
        const AA_TO_BOHR: f64 = 1.0e-10 / physical_constants::BOHR_RADIUS;
        geom_matr.mapv_inplace(|x| x * AA_TO_BOHR);

        Ok((z_vals, geom_matr, no_atoms))
    }

    #[inline(always)]
    fn no_atoms(&self) -> usize {
        self.atoms.len()
    }

    fn calc_core_potential(&self) -> f64 {
        let mut core_potential = 0.0;
        let coords = &self.geom.coords_matr;

        for i in 0..self.no_atoms {
            let r_i = coords.slice(s![i, ..]);
            for j in i + 1..self.no_atoms {
                let r_j = coords.slice(s![j, ..]);

                let r_ij = &r_i - &r_j;
                let r_ij_norm = r_ij.dot(&r_ij).sqrt();

                core_potential += (self.z_vals[i] as f64) * (self.z_vals[j] as f64) / r_ij_norm;
            }
        }
        core_potential
    }

    fn calc_core_potential_der(&self, deriv_atom: &Atom, cc: Cartesian) -> f64 {
        let mut core_potential_der = 0.0;

        for other_atom in self.atoms.iter() {
            if other_atom == deriv_atom {
                continue;
            }
            let r_ij_norm = deriv_atom - other_atom;
            let z_i = deriv_atom.z_val;
            let z_j = other_atom.z_val;
            match cc {
                Cartesian::X => {
                    core_potential_der += z_i as f64 * z_j as f64 * (deriv_atom[0] - other_atom[0])
                        / r_ij_norm.powi(3);
                }
                Cartesian::Y => {
                    core_potential_der += z_i as f64 * z_j as f64 * (deriv_atom[1] - other_atom[1])
                        / r_ij_norm.powi(3);
                }
                Cartesian::Z => {
                    core_potential_der += z_i as f64 * z_j as f64 * (deriv_atom[2] - other_atom[2])
                        / r_ij_norm.powi(3);
                }
            }
        }
        core_potential_der
    }

    fn calc_centre_of_mass(&self) {
        let mut COM = Array1::<f64>::zeros(3);
        for atom in self.atoms.iter() {
            COM[0] += atom.mass * atom[0]; // x
            COM[1] += atom.mass * atom[1]; // y
            COM[2] += atom.mass * atom[2]; // z
        }
        COM /= self.tot_mass;
    }

    fn other_two_idx(inp: usize) -> (usize, usize) {
        match inp {
            0 => (1, 2),
            1 => (0, 2),
            2 => (0, 1),
            _ => panic!("inp must be 0, 1, or 2"),
        }
    }

    fn calc_inertia_matr(&self) -> Array2<f64> {
        let mut inertia_matr = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            let (k, l) = Self::other_two_idx(i);
            for atom in self.atoms.iter() {
                inertia_matr[(i, i)] += atom.mass * (atom[k].powi(2) + atom[l].powi(2));
                for j in (i + 1)..3 {
                    inertia_matr[(i, j)] -= atom.mass * atom[i] * atom[j];
                    inertia_matr[(j, i)] = inertia_matr[(i, j)];
                }
            }
        }
        inertia_matr
    }

    /// Source: https://pythoninchemistry.org/ch40208/comp_chem_methods/moments_of_inertia.html
    fn mol_reorient_to_princ_ax_of_inertia(&mut self) {
        let mol_inertia_matr = self.calc_inertia_matr();
        let transform_matr = mol_inertia_matr
            .eigh(UPLO::Upper)
            .unwrap()
            .1
            .invh()
            .unwrap();

        // Debug print
        // println!("transform_matr:\n {}", transform_matr);
        // println!("self.geom.coords_matr (before):\n {}", self.geom.coords_matr);
        
        for mut row in self.geom.coords_matr.axis_iter_mut(Axis(0)) {
            let temp = transform_matr.dot(&row);
            row.assign(&temp);
        }
        // println!("self.geom.coords_matr (after):\n {}", self.geom.coords_matr);

        
        //* Update atom coords
        for (at_idx, atom) in self.atoms.iter_mut().enumerate() {
            atom[0] = self.geom.coords_matr[(at_idx, 0)];
            atom[1] = self.geom.coords_matr[(at_idx, 1)];
            atom[2] = self.geom.coords_matr[(at_idx, 2)];
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::relative_eq;

    use super::*;
    const WATER_90_FPATH: &str = "data/xyz/water90.xyz";

    #[test]
    fn test_mol_create() {
        let _test_mol = Molecule::new(WATER_90_FPATH, 0);
        println!("test_mol: {:?}", _test_mol);
    }

    #[test]
    fn test_no_atoms() {
        let test_mol = Molecule::new(WATER_90_FPATH, 0);
        let no_atoms = test_mol.no_atoms();
        assert_eq!(no_atoms, 3); // test case for water
    }

    #[test]
    #[should_panic]
    fn test_wrong_atom_in_input() {
        let wrong_inp_fpath = "data/xyz/wrong_input.xyz";
        let test_mol = Molecule::new(wrong_inp_fpath, 0);
        let no_atoms = test_mol.no_atoms();
        assert_eq!(no_atoms, 3); // test case for water
    }

    #[test]
    fn test_enum_string() {
        const test_str: &str = "H";
        let test_enum = PseElemSym::from_str(test_str);
        assert_eq!(test_enum.unwrap(), PseElemSym::H);
    }

    #[test]
    fn test_calc_core_potential() {
        let test_mol = Molecule::new(WATER_90_FPATH, 0);
        let core_potential = test_mol.calc_core_potential();
        println!("core_potential: {}", core_potential);
        assert!(relative_eq!(core_potential, 9.209396009090517, epsilon = 1.0e-10));// test case for water
    }

    #[test]
    fn test_calc_inertia_matr() {
        let test_mol = Molecule::new(WATER_90_FPATH, 0);
        let inertia_matr = test_mol.calc_inertia_matr();
    }

    #[test]
    fn test_mol_reorient() {
        let mol_fpath = "data/xyz/acetone.xyz";
        // let mol_fpath = "data/xyz/water90.xyz";
        let mut test_mol = Molecule::new(mol_fpath, 0);
        test_mol.mol_reorient_to_princ_ax_of_inertia();
        // test_mol.mol_reorient_to_princ_ax_of_inertia();
        // let test_after_inertia_matr = test_mol.calc_inertia_matr();
        // println!("test_after_inertia_matr:\n {}", test_after_inertia_matr);
    }
}
