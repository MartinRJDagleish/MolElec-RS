use crate::{calc_type::CalcSettings, print_utils::{print_header_for_section, fmt_f64}};

pub(crate) fn print_scf_header_and_settings(calc_sett: &CalcSettings) {
    println!("{:=>35}", "");
    println!("{:^35}", "RHF SCF");
    println!("{:=>35}", "");

    println!("{:-20}", "");
    println!("SCF settings:");
    println!("  {:<20} {:>10e}","ΔE THRESH", calc_sett.e_diff_thrsh);
    println!("  {:<20} {:>10e}","RMS FPS THRESH", calc_sett.commu_conv_thrsh);
    println!("  {:<20} {:>10}", "Direct SCF", calc_sett.use_direct_scf);
    println!("  {:<20} {:>10}","Use DIIS", calc_sett.use_diis);
    println!("  {:<20} {:>10}","DIIS Settings", "");
    println!("    {:<20} {:>8}","DIIS MIN", calc_sett.diis_sett.diis_min);
    println!("    {:<20} {:>8}","DIIS MAX", calc_sett.diis_sett.diis_max);
    println!("{:-20}", "");
}
