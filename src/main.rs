#![allow(dead_code, clippy::upper_case_acronyms, non_snake_case)]
#[macro_use]
extern crate lazy_static;
extern crate ndarray;
extern crate openblas_src;


mod basisset;
mod calc_type;
mod mol_int_and_deriv;
mod molecule;
mod print_utils;

use crate::{print_utils::print_initial_header, calc_type::CalcType};
use basisset::BasisSet;
use molecule::Molecule;

fn main() {
    //##################################
    //###           HEADER           ###
    //##################################
    let mut exec_times = print_utils::ExecTimes::new();
    exec_times.start("Total");

    print_initial_header();

    exec_times.start("Molecule");
    let _mol = Molecule::new("data/xyz/water90.xyz", 0);
    println!("Molecule: {:?}", _mol);
    exec_times.stop("Molecule");

    exec_times.start("BasisSet");
    let _basis = BasisSet::new("STO-3G", &_mol);
    println!("BasisSet: {:?}", _basis);
    // println!("Molecule: {:?}", _basis);
    // println!("\n\n");
    // for shell in basis.shell_iter() {
    //     println!("Shell: {:?}\n", shell);
    // }
    exec_times.stop("BasisSet");

    //##################################
    //###           BODY             ###
    //##################################
    // Calculation type
    let calc_type = CalcType::RHF;


    exec_times.stop("Total");

    //##################################
    //###           FOOTER           ###
    //##################################
    exec_times.print();
}
