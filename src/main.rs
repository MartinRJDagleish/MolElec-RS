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

use crate::{calc_type::CalcType, print_utils::print_logo};
use basisset::BasisSet;
use calc_type::{DiisSettings, CalcSettings};
use molecule::Molecule;
use crate::calc_type::rhf::rhf_scf_normal;

fn main() {
    //##################################
    //###           HEADER           ###
    //##################################
    let mut exec_times = print_utils::ExecTimes::new();
    exec_times.start("Total");

    print_logo();

    exec_times.start("Molecule");
    let mol = Molecule::new("data/xyz/furan.xyz", 0);
    // let mol = Molecule::new("data/xyz/water90.xyz", 0);
    // let mol = Molecule::new("data/xyz/calicheamicin_tinker_std.xtbopt.xyz", 0);
    // println!("Molecule: {:?}", _mol);
    exec_times.stop("Molecule");

    exec_times.start("BasisSet");
    let basis = BasisSet::new("STO-3G", &mol);
    // println!("BasisSet: {:?}", _basis);
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
    //
    exec_times.start("RHF noDIIS");
    let _calc_type = CalcType::RHF;

    let calc_sett = CalcSettings {
        max_scf_iter: 100,
        e_diff_thrsh: 1e-10,
        commu_conv_thrsh: 1e-10,
        use_diis: true,
        use_direct_scf: false,
        diis_sett: DiisSettings {
            diis_min: 2,
            diis_max: 6,
        },
    };
    
    let _scf = rhf_scf_normal(calc_sett, &mut exec_times, &basis, &mol);
    exec_times.stop("RHF noDIIS");
    

    exec_times.stop("Total");

    //##################################
    //###           FOOTER           ###
    //##################################
    exec_times.print();
}
