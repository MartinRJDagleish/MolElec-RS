#[macro_use]
extern crate lazy_static;
extern crate ndarray;
extern crate openblas_src;

mod basisset;
mod mol_int_and_deriv;
mod molecule;
mod print_utils;

use crate::print_utils::print_prog_header;
use molecule::Molecule;

fn main() {
    //##################################
    //###           HEADER           ###
    //##################################
    let mut exec_times = print_utils::ExecTimes::new();
    exec_times.start("total");

    print_prog_header();

    exec_times.start("molecule");
    let mol = Molecule::new("data/xyz/water90.xyz", 0);
    exec_times.stop("molecule");

    println!("Molecule: {:?}", mol);
    exec_times.stop("total");

    //##################################
    //###           FOOTER           ###
    //##################################
    exec_times.print();
}
