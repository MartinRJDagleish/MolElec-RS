#[macro_use]
extern crate lazy_static;
extern crate ndarray;
extern crate openblas_src;


mod basisset;
mod molecule;
mod print_utils;

use molecule::Molecule;

fn main() {
    let mol = Molecule::new("data/xyz/water90.xyz", 0);
    println!("Molecule: {:?}", mol);
}
