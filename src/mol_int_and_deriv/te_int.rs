use crate::{
    basisset::{CGTO, PGTO},
    mol_int_and_deriv::{
        oe_int::calc_vec_P,
        recurrence_rel::{EHermCoeff3D, RHermAuxInt},
    },
    molecule::{
        atom::Atom,
        cartesian_comp::{CC_X, CC_Y, CC_Z},
    },
};
use ndarray::array;
use std::f64::consts::PI;

#[allow(non_snake_case)]
pub fn calc_ERI_int_cgto(cgto1: &CGTO, cgto2: &CGTO, cgto3: &CGTO, cgto4: &CGTO) -> f64 {
    let mut eri_int = 0.0_f64;
    // let mut tmp;
    let vec_BA = cgto1.centre_pos().calc_vec_to_atom(cgto2.centre_pos());
    let vec_DC = cgto3.centre_pos().calc_vec_to_atom(cgto4.centre_pos());

    let ang_mom_vec1 = cgto1.ang_mom_vec();
    let ang_mom_vec2 = cgto2.ang_mom_vec();
    let ang_mom_vec3 = cgto3.ang_mom_vec();
    let ang_mom_vec4 = cgto4.ang_mom_vec();
    let ang_mom_vecs = [*ang_mom_vec1, *ang_mom_vec2, *ang_mom_vec3, *ang_mom_vec4];

    let t_max = ang_mom_vec1[CC_X] + ang_mom_vec2[CC_X];
    let u_max = ang_mom_vec1[CC_Y] + ang_mom_vec2[CC_Y];
    let v_max = ang_mom_vec1[CC_Z] + ang_mom_vec2[CC_Z];

    let tau_max = ang_mom_vec3[CC_X] + ang_mom_vec4[CC_X];
    let nu_max = ang_mom_vec3[CC_Y] + ang_mom_vec4[CC_Y];
    let phi_max = ang_mom_vec3[CC_Z] + ang_mom_vec4[CC_Z];
    let max_tuv_arr = [t_max, u_max, v_max, tau_max, nu_max, phi_max];

    let max_tot_ang_mom = *cgto1.ang_mom_type() as i32
        + *cgto2.ang_mom_type() as i32
        + *cgto3.ang_mom_type() as i32
        + *cgto4.ang_mom_type() as i32;

    let atoms = [
        cgto1.centre_pos(),
        cgto2.centre_pos(),
        cgto3.centre_pos(),
        cgto4.centre_pos(),
    ];
    //TODO: still wrong calculation
    for pgto1 in cgto1.pgto_iter() {
        for pgto2 in cgto2.pgto_iter() {
            for pgto3 in cgto3.pgto_iter() {
                for pgto4 in cgto4.pgto_iter() {
                    let pgtos = [pgto1, pgto2, pgto3, pgto4];
                    eri_int += pgto1.norm_const()
                        * pgto2.norm_const()
                        * pgto3.norm_const()
                        * pgto4.norm_const()
                        * pgto1.pgto_coeff()
                        * pgto2.pgto_coeff()
                        * pgto3.pgto_coeff()
                        * pgto4.pgto_coeff()
                        * calc_ERI_int_pgto(
                            pgtos,
                            atoms,
                            ang_mom_vecs,
                            max_tot_ang_mom,
                            &vec_BA,
                            &vec_DC,
                            &max_tuv_arr,
                        );
                }
            }
        }
    }
    eri_int
}

#[allow(non_snake_case)]
#[inline(always)]
pub fn calc_ERI_int_pgto(
    pgtos: [&PGTO; 4],
    atoms: [&&Atom; 4],
    ang_mom_vecs: [[i32; 3]; 4],
    max_tot_ang_mom: i32,
    vec_BA: &[f64; 3],
    vec_DC: &[f64; 3],
    max_tuv_arr: &[i32; 6],
) -> f64 {
    let mut eri_pgto_val = 0.0_f64;

    lazy_static! {
        pub static ref ERI_PI_FAC: f64 = 2.0 * PI * PI * PI.sqrt(); // 2 * pi^(5/2)
    }

    let ang_mom_vec1 = ang_mom_vecs[0];
    let ang_mom_vec2 = ang_mom_vecs[1];
    let ang_mom_vec3 = ang_mom_vecs[2];
    let ang_mom_vec4 = ang_mom_vecs[3];

    let vec_P = calc_vec_P(pgtos[0].alpha(), pgtos[1].alpha(), atoms[0], atoms[1]);
    let vec_Q = calc_vec_P(pgtos[2].alpha(), pgtos[3].alpha(), atoms[2], atoms[3]);
    let vec_PQ = array![
        vec_P[CC_X] - vec_Q[CC_X],
        vec_P[CC_Y] - vec_Q[CC_Y],
        vec_P[CC_Z] - vec_Q[CC_Z]
    ];

    let p = pgtos[0].alpha() + pgtos[1].alpha();
    let q = pgtos[2].alpha() + pgtos[3].alpha();
    let new_alph = p * q / (p + q);

    let R_tuv = RHermAuxInt::new(max_tot_ang_mom, vec_PQ, new_alph);

    let E_ab = EHermCoeff3D::new(pgtos[0].alpha(), pgtos[1].alpha(), vec_BA);
    let E_cd = EHermCoeff3D::new(pgtos[2].alpha(), pgtos[3].alpha(), vec_DC);

    for tau in 0..=max_tuv_arr[3] {
        let E_cd_ij = E_cd
            .E_ij
            .calc_recurr_rel(ang_mom_vec3[CC_X], ang_mom_vec4[CC_X], tau, 0);
        for nu in 0..=max_tuv_arr[4] {
            let E_cd_kl = E_cd
                .E_kl
                .calc_recurr_rel(ang_mom_vec3[CC_Y], ang_mom_vec4[CC_Y], nu, 0);
            for phi in 0..=max_tuv_arr[5] {
                let E_cd_mn =
                    E_cd.E_mn
                        .calc_recurr_rel(ang_mom_vec3[CC_Z], ang_mom_vec4[CC_Z], phi, 0);
                let E_cd_prod = E_cd_ij * E_cd_kl * E_cd_mn;
                if E_cd_prod == 0.0 {
                    continue;
                }
                let min_fac = if (tau + nu + phi) % 2 == 0 { 1.0 } else { -1.0 };
                for t in 0..=max_tuv_arr[0] {
                    let E_ab_ij =
                        E_ab.E_ij
                            .calc_recurr_rel(ang_mom_vec1[CC_X], ang_mom_vec2[CC_X], t, 0);
                    for u in 0..=max_tuv_arr[1] {
                        let E_ab_kl =
                            E_ab.E_kl
                                .calc_recurr_rel(ang_mom_vec1[CC_Y], ang_mom_vec2[CC_Y], u, 0);
                        for v in 0..=max_tuv_arr[2] {
                            let E_ab_mn = E_ab.E_mn.calc_recurr_rel(
                                ang_mom_vec1[CC_Z],
                                ang_mom_vec2[CC_Z],
                                v,
                                0,
                            );
                            let E_ab_prod = E_ab_ij * E_ab_kl * E_ab_mn;
                            if E_ab_prod == 0.0 {
                                continue;
                            }
                            let R_recurr_val = R_tuv.calc_recurr_rel(t + tau, u + nu, v + phi, 0);
                            eri_pgto_val += min_fac * E_ab_prod * E_cd_prod * R_recurr_val
                        }
                    }
                }
            }
        }
    }

    let ERI_fac = *ERI_PI_FAC * (1.0 / (p * q * (p + q).sqrt()));
    eri_pgto_val * ERI_fac
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basisset::parser::AngMomChar;
    use crate::molecule::PseElemSym;
    use approx::assert_relative_eq;

    fn init_two_diff_cgtos<'a>(atom1: &'a Atom, atom2: &'a Atom) -> (CGTO<'a>, CGTO<'a>) {
        let ang_mom_vec1: [i32; 3] = [0, 0, 0];
        let L_tot = ang_mom_vec1.iter().sum::<i32>();
        let pgto_vec1: Vec<PGTO> = vec![
            PGTO::new(130.7093214, 0.1543289673, &ang_mom_vec1, L_tot),
            PGTO::new(23.80886605, 0.5353281423, &ang_mom_vec1, L_tot),
            PGTO::new(6.443608313, 0.4446345422, &ang_mom_vec1, L_tot),
        ];
        let pgto_vec1_len = pgto_vec1.len();

        let ang_mom_vec2: [i32; 3] = [0, 0, 0];
        let pgto_vec2 = vec![
            PGTO::new(5.033151319, -0.09996722919, &ang_mom_vec2, L_tot),
            PGTO::new(1.169596125, 0.3995128261, &ang_mom_vec2, L_tot),
            PGTO::new(0.38038896, 0.7001154689, &ang_mom_vec2, L_tot),
        ];
        let pgto_vec2_len = pgto_vec2.len();

        let cgto1 = CGTO::new(pgto_vec1, pgto_vec1_len, AngMomChar::S, ang_mom_vec1, atom1);
        let cgto2 = CGTO::new(pgto_vec2, pgto_vec2_len, AngMomChar::S, ang_mom_vec1, atom2);
        (cgto1, cgto2)
    }

    fn init_two_same_cgtos(atom: &Atom) -> (CGTO, CGTO) {
        let ang_mom_vec1: [i32; 3] = [0, 0, 0];
        let l_tot = ang_mom_vec1.iter().sum::<i32>();
        let pgto_vec: Vec<PGTO> = vec![
            PGTO::new(130.7093214, 0.1543289673, &ang_mom_vec1, l_tot),
            PGTO::new(23.80886605, 0.5353281423, &ang_mom_vec1, l_tot),
            PGTO::new(6.443608313, 0.4446345422, &ang_mom_vec1, l_tot),
        ];
        let pgto_vec_len = pgto_vec.len();

        let cgto1 = CGTO::new(
            pgto_vec.clone(),
            pgto_vec_len,
            AngMomChar::S,
            ang_mom_vec1,
            atom,
        );
        let cgto2 = CGTO::new(
            pgto_vec.clone(),
            pgto_vec_len,
            AngMomChar::S,
            ang_mom_vec1,
            atom,
        );
        (cgto1, cgto2)
    }

    #[test]
    fn test_calc_eri_cgto_test1() {
        let atom = Atom::new(0.0, 0.0, 0.0, 8, crate::molecule::PseElemSym::O);
        let (cgto1, cgto2) = init_two_same_cgtos(&atom);
        let (cgto3, cgto4) = init_two_same_cgtos(&atom);

        let eri_val = calc_ERI_int_cgto(&cgto1, &cgto2, &cgto3, &cgto4);
        println!("ERI val: {}", eri_val);
        const ERI_REF_VAL1: f64 = 4.785065751815719;
        assert_relative_eq!(eri_val, ERI_REF_VAL1, epsilon = 1e-9);
    }
}
