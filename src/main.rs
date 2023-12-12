#![allow(dead_code, clippy::upper_case_acronyms)]
#[macro_use]
extern crate lazy_static;
extern crate ndarray;
extern crate openblas_src;

mod basisset;
mod mol_int_and_deriv;
mod molecule;
mod print_utils;
mod calc_type;

use crate::print_utils::print_prog_header;
use basisset::BasisSet;
use molecule::Molecule;

fn main() {
    //##################################
    //###           HEADER           ###
    //##################################
    let mut exec_times = print_utils::ExecTimes::new();
    exec_times.start("Total");

    print_prog_header();

    exec_times.start("Molecule");
    let _mol = Molecule::new("data/xyz/water90.xyz", 0);
    println!("Molecule: {:?}", _mol);
    exec_times.stop("Molecule");

    exec_times.start("BasisSet");
    let basis = BasisSet::new("STO-3G", &_mol);
    // println!("Molecule: {:?}", _basis);
    println!("\n\n");
    for shell in basis.shell_iter() {
        println!("Shell: {:?}\n", shell);
    }
    exec_times.stop("BasisSet");

    //##################################
    //###           BODY             ###
    //##################################
    // Calculation type
    // let calc_type = "";
    



    exec_times.stop("Total");

    //##################################
    //###           FOOTER           ###
    //##################################
    exec_times.print();
}
