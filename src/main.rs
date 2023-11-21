// use ndarray::prelude::*;
use molecule::Molecule;

// fn print_hello(inp_var: &str) {
//     println!("Hello world from {}", inp_var);
// }

mod molecule;

fn main() {
    let my_mol = Molecule::new("data/xyz/water90.xyz", 0);
    println!("Molecule: {:?}", my_mol);
}

