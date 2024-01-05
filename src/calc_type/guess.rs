#![allow(clippy::upper_case_acronyms, non_snake_case, dead_code)]

// Superposition of Atmoic Densities (SAD) guess implementation
// Source of implementation: Libint
// https://github.com/evaleev/libint/blob/4482d01abc37732a24b24799d3efe1d1f89a2891/include/libint2/chemistry/sto3g_atomic_density.h
//

#[inline(always)]
pub const fn sad_calc_num_ao(z_val: u32) -> u32 {
    match z_val {
        1 | 2 => 1,    // H & He
        3..=10 => 5,   // Li - Ne (2p incl. even for Li and Be)
        11..=18 => 9,  // Na - Ar (3p incl. even for Na and Mg)
        19..=20 => 13, // K & Ca (4p incl. even for K and Ca)
        21..=36 => 18, // Sc - Kr
        37..=38 => 22, // Rb & Sr (5p incl. even for Rb and Sr)
        39..=53 => 27, // Y - I
        _ => todo!(),
    }
}

// fn calc_occup_vec(z_val: u32) -> Vec<f64> {
//     let num_ao = sad_calc_num_ao(z_val) as usize;
//     let mut occ_vec = vec![0.0; num_ao];
//
//     let mut no_elec = z_val as usize;
//     match z_val {
//         1 | 2 => calc_subsh_occup_vec(&mut occ_vec, &mut no_elec),
//         3..=10 => {
//             calc_subsh_occup_vec(&mut occ_vec[0..1], &mut no_elec);
//             calc_subsh_occup_vec(&mut occ_vec[1..2], &mut no_elec);
//             calc_subsh_occup_vec(&mut occ_vec[2..], &mut no_elec);
//         }
//         11..=17 => {
//             calc_subsh_occup_vec(&mut occ_vec[0..1], &mut no_elec);
//             calc_subsh_occup_vec(&mut occ_vec[1..2], &mut no_elec);
//             calc_subsh_occup_vec(&mut occ_vec[2..=4], &mut no_elec);
//             calc_subsh_occup_vec(&mut occ_vec[5..], &mut no_elec);
//         }
//         _ => todo!(),
//     }
//
//     occ_vec
// }

fn calc_occup_vec(z_val: u32) -> Vec<f64> {
    let num_ao = sad_calc_num_ao(z_val) as usize;
    let mut occ_vec = vec![0.0; num_ao];

    let mut no_elec = z_val as usize;
    match z_val {
        1 | 2 => calc_subsh_occup_vec(&mut occ_vec, &mut no_elec),
        3..=10 => {
            calc_subsh_occup_vec(&mut occ_vec[0..1], &mut no_elec); // 1s
            calc_subsh_occup_vec(&mut occ_vec[1..2], &mut no_elec); // 2s
            calc_subsh_occup_vec(&mut occ_vec[2..], &mut no_elec); // 2p
        }
        11..=17 => {
            calc_subsh_occup_vec(&mut occ_vec[0..1], &mut no_elec); // 1s
            calc_subsh_occup_vec(&mut occ_vec[1..2], &mut no_elec); // 2s
            calc_subsh_occup_vec(&mut occ_vec[2..=4], &mut no_elec); // 2p
            calc_subsh_occup_vec(&mut occ_vec[5..6], &mut no_elec); // 3s
            calc_subsh_occup_vec(&mut occ_vec[6..], &mut no_elec); // 3p
        }
        18..=36 => {
            calc_subsh_occup_vec(&mut occ_vec[0..1], &mut no_elec); // 1s
            calc_subsh_occup_vec(&mut occ_vec[1..2], &mut no_elec); // 2s
            calc_subsh_occup_vec(&mut occ_vec[2..=4], &mut no_elec); // 2p
            calc_subsh_occup_vec(&mut occ_vec[5..6], &mut no_elec); // 3s
            calc_subsh_occup_vec(&mut occ_vec[6..=8], &mut no_elec); // 3p
            let mut no_4s_elec = if z_val == 19 || z_val == 24 || z_val == 29 {
                1
            } else {
                2
            };
            no_elec -= no_4s_elec;
            calc_subsh_occup_vec(&mut occ_vec[9..10], &mut no_4s_elec); // 4s

            let mut no_4p_elec = 6.min(if z_val > 30 { z_val - 30 } else { 0 }) as usize;
            calc_subsh_occup_vec(&mut occ_vec[10..=12], &mut no_4p_elec); // 4p
            calc_subsh_occup_vec(&mut occ_vec[13..], &mut no_elec) // 3d
        }
        _ => todo!(),
    }

    occ_vec
}

fn calc_subsh_occup_vec(occvec: &mut [f64], no_elec: &mut usize) {
    let size = occvec.len();
    let no_elec_alloc = (*no_elec).min(2 * size);
    *no_elec -= no_elec_alloc;
    let ne_per_orb = if no_elec_alloc % size == 0 {
        (no_elec_alloc / size) as f64
    } else {
        no_elec_alloc as f64 / size as f64
    };

    for f in occvec.iter_mut() {
        *f = ne_per_orb;
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use super::*;

    #[test]
    fn test_sad_calc_num_ao() {
        assert_eq!(sad_calc_num_ao(1), 1);
        assert_eq!(sad_calc_num_ao(2), 1);
        assert_eq!(sad_calc_num_ao(3), 5);
        assert_eq!(sad_calc_num_ao(10), 5);
        assert_eq!(sad_calc_num_ao(11), 9);
        assert_eq!(sad_calc_num_ao(18), 9);
        assert_eq!(sad_calc_num_ao(19), 13);
        assert_eq!(sad_calc_num_ao(20), 13);
        assert_eq!(sad_calc_num_ao(21), 18);
        assert_eq!(sad_calc_num_ao(36), 18);
        assert_eq!(sad_calc_num_ao(37), 22);
        assert_eq!(sad_calc_num_ao(38), 22);
        assert_eq!(sad_calc_num_ao(39), 27);
        assert_eq!(sad_calc_num_ao(53), 27);
    }

    #[test]
    fn test_calc_occup_vec() {
        let z_val: u32 = 24;
        let occup_vec = calc_occup_vec(z_val);
        const ORB_NAMES: [&str; 18]  = [
            "1s", "2s", "2px", "2py", "2pz", "3s", "3px", "3py", "3pz", "4s", "4px", "4py", "4pz",
            "3dxz", "3dyz", "3dz2", "3dx2y2", "3dxy", 
        ];
        println!("Z_val: {}", z_val);
        for (val, orb_name) in zip(occup_vec.iter(), ORB_NAMES.iter()) {
            println!("{:>5}: {:>10.5}", orb_name, val);
        }
        // println!("{:?}", occup_vec);
    }
    // #[test]
    // fn test_calc_subsh_occup_vec() {
    //     let mut test_vec = vec![1.2, 2.34, 234.1];
    //     let size = 25;
    //     let mut no_elec = 10;
    //     println!("{:?}", no_elec);
    //     calc_subsh_occup_vec(&mut test_vec, size, &mut no_elec);
    //     println!("{:?}", no_elec);
    //     calc_subsh_occup_vec(&mut test_vec, size, &mut no_elec);
    //     println!("{:?}", no_elec);
    //     calc_subsh_occup_vec(&mut test_vec, size, &mut no_elec);
    //     println!("{:?}", no_elec);
    // }
}
