mod basisset;
mod molecule;

// use ndarray::prelude::*;
use molecule::Molecule;

fn main() {
    let my_mol = Molecule::new("data/xyz/water90.xyz", 0);
    println!("Molecule: {:?}", my_mol);
}
