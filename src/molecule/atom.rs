use crate::molecule::PSE_ELEM_SYMS_STR;
use std::ops::{Index, IndexMut, Sub};

#[rustfmt::skip]
const ATOMIC_MASSES_IN_AMU: [f64; 117] = [
    1.00782503223, 4.00260325413, 7.0160034366, 9.012183065, 11.00930536, 12.0000000, 14.00307400443, 15.99491461957, 18.99840316273, 19.9924401762, 22.9897692820,
    23.985041697, 26.98153853, 27.97692653465, 30.97376199842, 31.9720711744, 34.968852682, 39.9623831237, 38.9637064864, 39.962590863, 44.95590828, 47.94794198, 50.94395704,
    51.94050623, 54.93804391, 55.93493633, 58.93319429, 57.93534241, 62.92959772, 63.92914201, 68.9255735, 73.921177761, 74.92159457, 79.9165218, 78.9183376, 83.9114977282, 84.9117897379,
    87.9056125, 88.9058403, 89.9046977, 92.9063730, 97.90540482, 97.9072124, 101.9043441, 102.9054980, 105.9034804, 106.9050916, 113.90336509, 114.903878776, 119.90220163,
    120.9038120, 129.906222748, 126.9044719, 131.9041550856, 132.9054519610, 137.90524700, 138.9063563, 139.9054431, 140.9076576, 141.9077290, 144.9127559, 151.9197397,
    152.9212380, 157.9241123, 158.9253547, 163.9291819, 164.9303288, 165.9302995, 168.9342179, 173.9388664, 174.9407752, 179.9465570, 180.9479958, 183.95093092, 186.9557501,
    191.9614770, 192.9629216, 194.9647917, 196.96656879, 201.97064340, 204.9744278, 207.9766525, 208.9803991, 208.9824308, 209.9871479, 222.0175782, 223.0197360, 226.0254103,
    227.0277523, 232.0380558, 231.0358842, 238.0507884, 237.0481736, 244.0642053, 243.0613813, 247.0703541, 247.0703073, 251.0795886, 252.082980, 257.0951061, 258.0984315,
    259.10103, 266.11983, 267.12179, 268.12567, 271.13393, 270.13336, 269.13375, 278.15631, 281.16451, 282.16912, 285.17712, 286.18221, 289.19042, 289.19363, 293.20449, 294.21046,
];

#[derive(Debug, Default, PartialEq)]
pub struct Atom {
    x: f64,
    y: f64,
    z: f64,
    pub z_val: u32,
}

impl Index<usize> for Atom {
    type Output = f64; // necessary for Index trait
    fn index<'a>(&'a self, i: usize) -> &'a f64 {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Index out of bounds for Atom"),
        }
    }
}

impl IndexMut<usize> for Atom {
    fn index_mut<'a>(&'a mut self, i: usize) -> &'a mut f64 {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Index out of bounds for Atom"),
        }
    }
}

impl Sub for Atom {
    type Output = f64;
    fn sub(self, other: Self) -> Self::Output {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

impl<'a> Sub for &'a Atom {
    type Output = f64;

    fn sub(self, other: Self) -> Self::Output {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}


impl Atom {
    pub fn new(x_inp: f64, y_inp: f64, z_inp: f64, z_val: u32) -> Self {
        Self {
            x: x_inp,
            y: y_inp,
            z: z_inp,
            z_val,
        }
    }

    fn to_sym_str(&self) -> String {
        PSE_ELEM_SYMS_STR[self.z_val as usize].to_string()
    }

    pub fn z_val_to_sym_str(z_val: usize) -> String {
        PSE_ELEM_SYMS_STR[z_val].to_string()
    }
}

// impl IndexMut<usize> for Atom {
//     fn index_mut<'a>(&'a mut self, i: usize) -> &'a mut f32 {
//         &mut self.e[i]
//     }
// }
