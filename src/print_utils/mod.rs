use std::{collections::HashMap, time::Instant};

pub mod print_scf;

pub fn print_header_logo() {
    //     const HEADER_V1: &str = r#"
    // ___  ___      _   _____ _            ______  _____
    // |  \/  |     | | |  ___| |           | ___ \/  ___|
    // | .  . | ___ | | | |__ | | ___  ___  | |_/ /\ `--.
    // | |\/| |/ _ \| | |  __|| |/ _ \/ __| |    /  `--. \
    // | |  | | (_) | | | |___| |  __/ (__  | |\ \ /\__/ /
    // \_|  |_/\___/|_| \____/|_|\___|\___| \_| \_|\____/
    // "#;
    //
    //     const HEADER_V2: &str = r#"
    //      __  ___      __   ________          ____  _____
    //     /  |/  /___  / /  / ____/ /__  _____/ __ \/ ___/
    //    / /|_/ / __ \/ /  / __/ / / _ \/ ___/ /_/ /\__ \
    //   / /  / / /_/ / /  / /___/ /  __/ /__/ _, _/___/ /
    //  /_/  /_/\____/_/  /_____/_/\___/\___/_/ |_|/____/
    // "#;
    //
    //     const HEADER_V3: &str = r#"
    //      __  ___      __            ________          ____  _____
    //     /  |/  /___  / /           / ____/ /__  _____/ __ \/ ___/
    //    / /|_/ / __ \/ /  ______   / __/ / / _ \/ ___/ /_/ /\__ \
    //   / /  / / /_/ / /  /_____/  / /___/ /  __/ /__/ _, _/___/ /
    //  /_/  /_/\____/_/           /_____/_/\___/\___/_/ |_|/____/
    //     "#;
    //
    //     const HEADER_V4: &str = r#"
    //     888b     d888          888        8888888888 888                   8888888b.   .d8888b.
    //     8888b   d8888          888        888        888                   888   Y88b d88P  Y88b
    //     88888b.d88888          888        888        888                   888    888 Y88b.
    //     888Y88888P888  .d88b.  888        8888888    888  .d88b.   .d8888b 888   d88P  "Y888b.
    //     888 Y888P 888 d88""88b 888 888888 888        888 d8P  Y8b d88P"    8888888P"      "Y88b.
    //     888  Y8P  888 888  888 888        888        888 88888888 888      888 T88b         "888
    //     888   "   888 Y88..88P 888        888        888 Y8b.     Y88b.    888  T88b  Y88b  d88P
    //     888       888  "Y88P"  888        8888888888 888  "Y8888   "Y8888P 888   T88b  "Y8888P"
    //     "#;
    //
    // println!("{}", HEADER_V1);
    // println!("{}", HEADER_V2);
    // println!("{}", HEADER_V3);
    // println!("{}", HEADER_V4);

    const HEADER_V5: &str = r#"
  ,ggg, ,ggg,_,ggg,                                   ,ggggggg,                          ,ggggggggggg,         ,gg,   
 dP""Y8dP""Y88P""Y8b               ,dPYb,           ,dP""""""Y8b ,dPYb,                 dP"""88""""""Y8,      i8""8i  
 Yb, `88'  `88'  `88               IP'`Yb           d8'    a  Y8 IP'`Yb                 Yb,  88      `8b      `8,,8'  
  `"  88    88    88               I8  8I           88     "Y8P' I8  8I                  `"  88      ,8P       `88'   
      88    88    88               I8  8'           `8baaaa      I8  8'                      88aaaad8P"        dP"8,  
      88    88    88    ,ggggg,    I8 dP  aaaaaaaa ,d8P""""      I8 dP   ,ggg,     ,gggg,    88""""Yb,        dP' `8a 
      88    88    88   dP"  "Y8ggg I8dP   """""""" d8"           I8dP   i8" "8i   dP"  "Yb   88     "8b      dP'   `Yb
      88    88    88  i8'    ,8I   I8P             Y8,           I8P    I8, ,8I  i8'         88      `8i _ ,dP'     I8
      88    88    Y8,,d8,   ,d8'  ,d8b,_           `Yba,,_____, ,d8b,_  `YbadP' ,d8,_    _   88       Yb,"888,,____,dP
      88    88    `Y8P"Y8888P"    8P'"Y88            `"Y8888888 8P'"Y88888P"Y888P""Y8888PP   88        Y8a8P"Y88888P" 
"#;

    println!("{}", HEADER_V5);
}

const HEADER_STR: &str = "**********************************************************************";

pub(crate) fn print_header_for_section(inp_str: &str) {
    let centered_str = format!("*{:^68}*", inp_str);
    println!("\n{}\n{}\n{}\n", HEADER_STR, centered_str, HEADER_STR);
}

/// ## Format a floating point number in scientific notation
/// - `width` controls the amount of left padded spaces
/// - `precision` is the amount of decimals
/// - `exp_pad` controls the amount of left padded 0s
///
/// Source: https://stackoverflow.com/questions/65264069/alignment-of-floating-point-numbers-printed-in-scientific-notation
///
pub(crate) fn fmt_f64(num: f64, width: usize, precision: usize, exp_pad: usize) -> String {
    let mut num = format!("{:.precision$e}", num, precision = precision);
    // Safe to `unwrap` as `num` is guaranteed to contain `'e'`
    let exp = num.split_off(num.find('e').unwrap());

    let (sign, exp) = if exp.starts_with("e-") {
        // ('-', &exp[2..])
        ('-', exp.strip_prefix("e-").unwrap())
    } else {
        ('+', exp.strip_prefix('e').unwrap())
    };
    num.push_str(&format!("e{}{:0>pad$}", sign, exp, pad = exp_pad));

    format!("{:>width$}", num, width = width)
}

pub struct ExecTimes {
    timings_map: HashMap<String, [Instant; 2]>,
}

impl ExecTimes {
    pub fn new() -> Self {
        Self {
            timings_map: HashMap::new(),
        }
    }

    pub fn start(&mut self, name: &str) {
        let now = Instant::now();
        self.timings_map.insert(name.to_string(), [now, now]);
    }

    pub fn stop(&mut self, name: &str) {
        let now = Instant::now();
        if let Some(timings) = self.timings_map.get_mut(name) {
            timings[1] = now;
        }
    }

    pub fn print_wo_order(&self) {
        print_header_for_section("Execution times");
        for (name, timings) in &self.timings_map {
            if name == "Total" {
                continue;
            }
            let exec_time = timings[1].duration_since(timings[0]);
            println!("{: >4}{:<25}{:>15?}", "", name, exec_time);
        }

        let tot_instants = self.timings_map.get("Total").unwrap();
        let tot_exec_time = tot_instants[1].duration_since(tot_instants[0]);
        println!("\n\n{: >4}{:<25}{:>15?}","",  "Total execution time", tot_exec_time);
    }
}
