use crate::calc_type::{CalcSettings, Reference};

pub(crate) fn print_scf_header_and_settings(calc_sett: &CalcSettings, calc_type: Reference) {
    println!("{:=>35}", "");
    match calc_type {
        Reference::RHF => println!("{:^35}", "RHF SCF"),
        Reference::UHF => println!("{:^35}", "UHF SCF"),
        Reference::ROHF => println!("{:^35}", "ROHF SCF"),
    }
    println!("{:=>35}", "");
    const SING_IND: &str = "  ";
    const DBL_IND: &str = "    ";

    // ↓ this is used to not format this part 
    #[allow(clippy::deprecated_cfg_attr)]
    #[cfg_attr(rustfmt, rustfmt_skip)] {
    println!("{:-20}", "");
    println!("SCF settings:");
    println!("{}{:<20} {:>10e}", SING_IND, "ΔE THRESH", calc_sett.e_diff_thrsh);
    println!("{}{:<20} {:>10e}", SING_IND, "RMS FPS THRESH", calc_sett.commu_conv_thrsh);
    println!("{}{:<20} {:>10}", SING_IND, "Direct SCF", calc_sett.use_direct_scf);
    println!("{}{:<20} {:>10}", SING_IND, "Use DIIS", calc_sett.use_diis);
    println!("{}{:<20} {:>10}", SING_IND, "DIIS Settings", "");
    println!("{}{:<20} {:>8}", DBL_IND, "DIIS MIN", calc_sett.diis_sett.diis_min);
    println!("{}{:<20} {:>8}", DBL_IND, "DIIS MAX", calc_sett.diis_sett.diis_max);
    println!("{:-20}", "");
    }
}
