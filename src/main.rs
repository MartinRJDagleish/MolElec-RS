// #![allow(dead_code, clippy::upper_case_acronyms, non_snake_case)]
#![allow(clippy::upper_case_acronyms, non_snake_case)]
#[macro_use]
extern crate lazy_static;
extern crate ndarray;
extern crate openblas_src;

mod basisset;
mod calc_type;
mod mol_int_and_deriv;
mod molecule;
mod print_utils;

use crate::{calc_type::HF_Ref, print_utils::print_header_logo};
use basisset::BasisSet;
use calc_type::{rhf::RHF, uhf::UHF, CalcSettings, DiisSettings, HF};
use molecule::Molecule;

fn main() {
    //##################################
    //###           HEADER           ###
    //##################################
    let mut exec_times = print_utils::ExecTimes::new();
    exec_times.start("Total");

    print_header_logo();

    exec_times.start("Molecule");
    let mol = Molecule::new("data/xyz/water90.xyz", 0);
    // let mol = Molecule::new("data/xyz/furan.xyz", 0);
    // let mol = Molecule::new("data/xyz/calicheamicin_tinker_std.xtbopt.xyz", 0);
    // println!("Molecule: {:?}", _mol);
    exec_times.stop("Molecule");

    exec_times.start("BasisSet");
    let basis = BasisSet::new("STO-3G", &mol);
    // let basis = BasisSet::new("6-311++G**", &mol);
    exec_times.stop("BasisSet");

    //##################################
    //###           BODY             ###
    //##################################
    // Calculation type
    //
    let _calc_type = HF_Ref::RHF_ref;

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

    exec_times.start("RHF DIIS indir");
    let mut rhf = RHF::new(&basis, &calc_sett);
    let _scf = rhf.run_scf(&calc_sett, &mut exec_times, &basis, &mol);
    exec_times.stop("RHF DIIS indir");
    exec_times.start("UHF DIIS indir");
    let mut uhf: UHF = UHF::new(&basis, &calc_sett);
    let _scf = uhf.run_scf(&calc_sett, &mut exec_times, &basis, &mol);
    // let _scf = uhf_scf_normal(&calc_sett, &mut exec_times, &basis, &mol);
    exec_times.stop("UHF DIIS indir");

    exec_times.stop("Total");

    //##################################
    //###           FOOTER           ###
    //##################################
    exec_times.print_wo_order();
}
