// use ndarray::prelude::*;
use molecule::Molecule;

// fn print_hello(inp_var: &str) {
//     println!("Hello world from {}", inp_var);
// }

mod molecule;

fn main() {
    let my_mol = Molecule::new("data/xyz/water90.xyz");
    println!("Molecule: {:?}", my_mol);
    // let mat_a = Array1::from(vec![1., 2., 3., 4.]);
    // println!("mat_a: {:?}", mat_a);
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mol_create() {
        let water_90_fpath = "data/xyz/water90.xyz";
        let test_mol = Molecule::new(water_90_fpath);
        // assert_eq!(test_mol)
        // print_hello(input);
    }
}