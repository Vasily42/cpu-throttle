extern crate clap;
extern crate ctrlc;
extern crate is_superuser;
extern crate mathru;
extern crate nix;
extern crate num_cpus;
extern crate posixmq;
extern crate text_io;

use clap::{Parser, Subcommand};
use core::{f64, time::Duration};
use std::{
    fs::Permissions,
    i32,
    os::unix::fs::PermissionsExt,
    path::Path,
    process::Command,
    str::{from_utf8, FromStr},
    sync::{
        atomic::{
            AtomicBool, AtomicI32,
            Ordering::{self, *},
        },
        Arc, LazyLock,
    },
};

const MAX_IDLENESS: u8 = 30;
const MIN_DISCRT_PERIOD_MS: i32 = 150;
const MAX_DISCRT_PERIOD_MS: i32 = 4000;
const THROTTLING_START_TIME_MS: i32 = 7000;
const THROTTLING_RELEASE_TIME_MS: i32 = 12000;

static N_CPUS: LazyLock<i32> = LazyLock::new(|| num_cpus::get() as i32);

static MAX_CPU_FREQ: LazyLock<i32> =
    LazyLock::new(|| read_i32("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"));

static MIN_CPU_FREQ: LazyLock<i32> =
    LazyLock::new(|| read_i32("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq"));

static TEMPERATURE_PROVIDER_FILE: LazyLock<String> = LazyLock::new(|| {
    let mut find_cmd_output = Command::new("find")
        .args(["/sys", "-name", "temp*_input"])
        .output()
        .expect("Error on find command")
        .stdout;

    let mut temper_file_list: Vec<(&str, i32)> = std::str::from_utf8_mut(&mut find_cmd_output)
        .unwrap()
        .split("\n")
        .filter(|s| !s.is_empty())
        .map(|s| (s, read_i32(s)))
        .collect();

    temper_file_list.sort_by_key(|pair| pair.1);

    temper_file_list
        .last()
        .expect("There are no hwmon files")
        .0
        .to_string()
});

static CPU_IDLENESS: LazyLock<std::sync::RwLock<Vec<u8>>> =
    LazyLock::new(|| std::sync::RwLock::new(vec![MAX_IDLENESS; *N_CPUS as usize]));

static STANDART_MULTIPLIER: LazyLock<AtomicI32> = LazyLock::new(|| {
    AtomicI32::new(
        match std::fs::read_to_string("/var/lib/cpu-throttle/opt_multiplier_value") {
            Ok(s) => s.trim().parse::<i32>().unwrap(),
            Err(_) => 150,
        },
    )
});

static CLAMP_MIN_CPU_FREQ: LazyLock<i32> =
    LazyLock::new(
        || match std::fs::read_to_string("/var/lib/cpu-throttle/clamp_min_freq") {
            Ok(s) => s.trim().parse::<i32>().unwrap(),
            Err(_) => *MIN_CPU_FREQ,
        },
    );

static DISCRT_PERIOD_MS: LazyLock<AtomicI32> = LazyLock::new(|| AtomicI32::new(MIN_DISCRT_PERIOD_MS));

static MQUEUE: LazyLock<posixmq::PosixMq> = LazyLock::new(|| get_mqueue());

struct FrequencyController {
    target_t: i32,
    prev_t: i32,
    temp_velocity_err: f64,
    grad_multiplier: i32,
}

impl FrequencyController {
    fn new(target_temperature: i32) -> Self {
        Self {
            target_t: target_temperature,
            prev_t: get_temp(),
            temp_velocity_err: 0.0,
            grad_multiplier: STANDART_MULTIPLIER.load(Relaxed),
        }
    }

    fn get_delta_freq(&mut self, t: i32) -> i32 {
        let current_t = t;

        let dt = current_t - self.prev_t;
        self.prev_t = current_t;

        let temp_velocity = dt as f64 / DISCRT_PERIOD_MS.load(Relaxed) as f64;

        let proportional_temp_diff = (current_t - self.target_t) as f64 / 1000.0;
        let target_temp_velocity_curve = if proportional_temp_diff > 0.0 {
            2.0 * ((-0.5 * proportional_temp_diff).exp() - 1.0)
        } else {
            -proportional_temp_diff
        };

        self.temp_velocity_err = temp_velocity - target_temp_velocity_curve;

        let grad = self.grad_multiplier as f64 * 1000.0 * self.temp_velocity_err;

        grad as i32
    }
}

#[derive(Subcommand, PartialEq, Clone, Copy)]
enum InterThreadMessage {
    /// Try to find better hyperparameters (highly recommended)
    Optimize,
    /// Pause throttling
    Pause,
    /// Continue throttling
    Continue,
    /// Pause/Continue throttling
    Toggle,
    /// Read new values from config files (for internal use mostly)
    ReadConfig,
    /// Exit all threads, finish with success code
    Exit,
    /// Set target temperature
    At(TempArg),
}

impl ToString for InterThreadMessage {
    fn to_string(&self) -> String {
        match self {
            InterThreadMessage::Optimize => String::from("optimize"),
            InterThreadMessage::Pause => String::from("pause"),
            InterThreadMessage::Continue => String::from("continue"),
            InterThreadMessage::Toggle => String::from("toggle"),
            InterThreadMessage::ReadConfig => String::from("read-config"),
            InterThreadMessage::Exit => String::from("exit"),
            InterThreadMessage::At(temperature) => temperature.temperature.to_string(),
        }
    }
}

impl FromStr for InterThreadMessage {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim_end_matches('\0').to_lowercase().as_str() {
            "optimize" => Ok(InterThreadMessage::Optimize),
            "pause" => Ok(InterThreadMessage::Pause),
            "continue" => Ok(InterThreadMessage::Continue),
            "toggle" => Ok(InterThreadMessage::Toggle),
            "read-config" => Ok(InterThreadMessage::ReadConfig),
            "exit" => Ok(InterThreadMessage::Exit),
            maybe_num => match maybe_num.parse::<i32>().ok() {
                Some(t) => Ok(InterThreadMessage::At(TempArg { temperature: t })),
                None => Err("There is no such signal".to_string()),
            },
        }
    }
}

#[derive(clap::Args, Clone, Copy, PartialEq)]
struct TempArg {
    temperature: i32,
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: InterThreadMessage,
}

fn main() -> Result<(), i32> {
    let args = Args::parse();

    if already_run() {
        use InterThreadMessage::*;
        match args.command {
            InterThreadMessage::Optimize => {
                if !is_superuser::is_superuser() {
                    eprintln!("Run this command [optimize] with sudo!");
                    return Err(1);
                }
                send_msg(Pause);
                optimize();
                send_msg(ReadConfig);
                send_msg(Continue);
            }
            _ => {
                send_msg(args.command);
            }
        };

        return Ok(());
    }

    if Path::new("/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies")
        .try_exists()
        .unwrap()
    {
        eprintln!("The CPU probably doesn't support fine-grained frequency scaling");
        return Err(1);
    }

    let target_temperature = match args.command {
        InterThreadMessage::At(temperature) => temperature.temperature,
        _ => {
            eprintln!("You must enter temperature");
            return Err(1);
        }
    };

    let target_t = Arc::new(AtomicI32::new(target_temperature * 1000));

    let limit_freq = match *N_CPUS {
        1 => limit_freq_uniform,
        2.. => limit_freq_multiform,
        ..=0 => {
            eprintln!("wtf");
            return Err(1);
        }
    };

    fast_unlock();

    let throttling = Arc::new(AtomicBool::new(true));
    let t_wait_term = throttling.clone();

    ctrlc::set_handler(move || {
        t_wait_term.store(false, Release);
    })
    .expect("Error setting SIGTERM handler");

    let mut freq_ctl = FrequencyController::new(target_t.load(Relaxed));
    let mut curr_freq: i32 = *MAX_CPU_FREQ;
    let mut paused = false;
    let mut overall_restlessness: f64 = 0.0;
    let mut clamp_min_cpu_freq = *CLAMP_MIN_CPU_FREQ;

    while throttling.load(Acquire) {
        std::thread::sleep(Duration::from_millis(DISCRT_PERIOD_MS.load(Relaxed) as u64));

        use InterThreadMessage::*;
        let msg = receive_msg();
        if let Some(msg) = msg {
            match msg {
                Pause | Continue | Toggle => {
                    if paused && (msg == Toggle || msg == Continue) {
                        paused = false;
                    } else if !paused && (msg == Toggle || msg == Pause) {
                        fast_unlock();
                        curr_freq = *MAX_CPU_FREQ;
                        overall_restlessness = 0.0;
                        DISCRT_PERIOD_MS.store(MAX_DISCRT_PERIOD_MS, Ordering::Relaxed);
                        paused = true;
                    }
                }
                ReadConfig => {
                    freq_ctl.grad_multiplier =
                        read_i32("/var/lib/cpu-throttle/opt_multiplier_value");
                    clamp_min_cpu_freq = read_i32("/var/lib/cpu-throttle/clamp_min_freq");
                }
                At(temperature) => {
                    freq_ctl.target_t = temperature.temperature * 1000;
                }
                Exit => {
                    break;
                }
                _ => {}
            }
        }

        if paused {
            continue;
        }

        let actual_t = get_temp();
        let delta_freq = freq_ctl.get_delta_freq(actual_t);

        if delta_freq > 0 && actual_t > target_t.load(Relaxed) - 5000 || curr_freq != *MAX_CPU_FREQ
        {
            overall_restlessness +=
                DISCRT_PERIOD_MS.load(Relaxed) as f64 / THROTTLING_START_TIME_MS as f64;
            if overall_restlessness > 1.0 {
                overall_restlessness = 1.0;
            }
        } else if curr_freq == *MAX_CPU_FREQ {
            overall_restlessness -=
                DISCRT_PERIOD_MS.load(Relaxed) as f64 / THROTTLING_RELEASE_TIME_MS as f64;
            if overall_restlessness < 0.0 {
                overall_restlessness = 0.0;
            }
        }

        let new_dscrt_period = (MAX_DISCRT_PERIOD_MS as f64 * (1.0 - overall_restlessness))
            .clamp(MIN_DISCRT_PERIOD_MS as f64, MAX_DISCRT_PERIOD_MS as f64)
            as i32;

        if overall_restlessness >= 0.9 {
            curr_freq -= delta_freq;
            curr_freq = curr_freq.clamp(clamp_min_cpu_freq, *MAX_CPU_FREQ);

            limit_freq(curr_freq);
        } else {
            if curr_freq != *MAX_CPU_FREQ {
                curr_freq = *MAX_CPU_FREQ;
                fast_unlock();
            }
            freq_ctl.prev_t = actual_t;
        }

        DISCRT_PERIOD_MS.store(new_dscrt_period, Ordering::Relaxed);
    }

    println!("exiting...");
    fast_unlock();

    posixmq::remove_queue("/cpu-throttle").expect("Cannot close message queue");
    Ok(())
}

fn get_mqueue() -> posixmq::PosixMq {
    let mq = posixmq::OpenOptions::readwrite()
        .mode(0o777)
        .nonblocking()
        .capacity(3)
        .max_msg_len(32)
        .create()
        .open("/cpu-throttle")
        .unwrap();

    if is_superuser::is_superuser() {
        std::fs::set_permissions("/dev/mqueue/cpu-throttle", Permissions::from_mode(0o777))
            .expect("Cannot set permissions");
    }
    mq
}

fn send_msg(msg: InterThreadMessage) {
    MQUEUE
        .send(0, msg.to_string().as_bytes())
        .expect("Cannot send message");
}

fn receive_msg() -> Option<InterThreadMessage> {
    if MQUEUE.attributes().unwrap().current_messages == 0 {
        return None;
    }
    let mut msg_buffer = [0_u8; 32];
    match MQUEUE.recv(&mut msg_buffer) {
        Ok(_) => Some(from_utf8(&msg_buffer).unwrap().parse().unwrap()),
        Err(_) => None,
    }
}

fn limit_freq_multiform(freq: i32) {
    let mut cpu_idleness = CPU_IDLENESS.write().unwrap();
    for i in 0..(*N_CPUS as usize) {
        let curr_freq = read_i32(&format!(
            "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq",
            i
        ));

        if curr_freq > *MIN_CPU_FREQ + ((freq - *MIN_CPU_FREQ) as f64 * 0.8) as i32 {
            cpu_idleness[i] = 0
        } else if curr_freq < *MIN_CPU_FREQ + ((freq - *MIN_CPU_FREQ) as f64 * 0.2) as i32 {
            cpu_idleness[i] += 1;
            cpu_idleness[i] = cpu_idleness[i].clamp(0, MAX_IDLENESS);
        }

        if cpu_idleness[i] >= MAX_IDLENESS - 2 {
            std::fs::write(
                format!("/sys/devices/system/cpu/cpu{}/cpufreq/scaling_max_freq", i),
                (*MAX_CPU_FREQ).to_string(),
            )
            .expect("Cannot write to /sys");
        } else {
            std::fs::write(
                format!("/sys/devices/system/cpu/cpu{}/cpufreq/scaling_max_freq", i),
                freq.to_string(),
            )
            .expect("Cannot write to /sys");
        }
    }
}

fn limit_freq_uniform(freq: i32) {
    for i in 0..*N_CPUS {
        std::fs::write(
            format!("/sys/devices/system/cpu/cpu{}/cpufreq/scaling_max_freq", i),
            freq.to_string(),
        )
        .expect("Cannot write to /sys");
    }
}

fn fast_unlock() {
    limit_freq_uniform(*MAX_CPU_FREQ);
}

fn get_temp() -> i32 {
    read_i32(&TEMPERATURE_PROVIDER_FILE)
}

fn read_i32(path: &str) -> i32 {
    let data = std::fs::read(path).expect(&format!("Cannot read from {}", path));
    let int_str = from_utf8(&data).unwrap();
    let trimmed = int_str.trim();
    let value = trimmed.parse().unwrap();

    value
}

fn already_run() -> bool {
    let mut pgrep = Command::new("pgrep");
    pgrep.arg("cpu-throttle");
    let output = pgrep.output().unwrap().stdout;
    let output_str = from_utf8(&output).unwrap();
    let lines: Vec<&str> = output_str.split('\n').collect();

    lines.len() > 2
}

fn optimize() {
    print!("Do you really want to find optimal hyperparameters (some of them)? [Y/n]: ");
    let input: String = text_io::read!("{}\n");
    if !input.is_empty() && !input.to_ascii_lowercase().starts_with('y') {
        println!("Ok, goodbye :)");
        return;
    }

    let optimal_multiplier_file = Path::new("/var/lib/cpu-throttle/opt_multiplier_value");
    let clamp_min_freq_file = Path::new("/var/lib/cpu-throttle/clamp_min_freq");
    if !optimal_multiplier_file.exists() {
        let parent_dir = optimal_multiplier_file.parent().unwrap();
        if !parent_dir.exists() {
            std::fs::DirBuilder::new()
                .create(parent_dir)
                .expect("failed creating parent dir in /var");
        }
    } else {
        print!("Optimal values were previosly established. Continue anyway? [Y/n]: ");
        let input: String = text_io::read!("{}\n");
        if !input.is_empty() && !input.to_ascii_lowercase().starts_with('y') {
            println!("Ok, goodbye ;)");
            return;
        }
    }

    print!("Enter the maximum CPU temperature you consider acceptable [85]: ");
    let target_t = 1000
        * loop {
            let input: String = text_io::read!("{}\n");
            if input.is_empty() {
                break 85;
            } else {
                match input.trim().parse() {
                    Ok(t) => {
                        if t > 120 || t < 20 {
                            print!("Temperature must be between 20 and 120 (Celsius) [85]: ");
                            continue;
                        } else {
                            break t;
                        }
                    }
                    Err(_) => {
                        print!("You need to enter an integer value between 20 and 120 [85]: ");
                        continue;
                    }
                }
            }
        };

    let mut freq_ctl = FrequencyController::new(target_t);
    let optimal_freq: Arc<AtomicI32> = Arc::new(AtomicI32::new(0));

    let mut test = |multiplier: i32| -> i64 {
        let stress_task = |sec: i32, load: f64| {
            let ld_cpus = ((*N_CPUS as f64 * load) as i32).clamp(1, *N_CPUS);
            let mut command = Command::new("timeout");
            command.args([&format!("{}s", sec), "stress", "-c", &ld_cpus.to_string()]);
            command.status().expect("Cannot execute stress command");
        };
        println!("Warming up...");
        stress_task(15, 1.0);

        freq_ctl.grad_multiplier = multiplier;

        let mut temperature_velocity_deviation_power = 0i64;

        let mut curr_freq = *MAX_CPU_FREQ;

        println!("testing with {} multiplier...", multiplier);

        let complex_stress_task = std::thread::spawn(move || {
            stress_task(5, 1.0);
            stress_task(3, 0.2);
            std::thread::sleep(Duration::from_secs(2));
            stress_task(1, 1.0);
            stress_task(1, 0.3);
            stress_task(1, 1.0);
            stress_task(1, 0.4);
            std::thread::sleep(Duration::from_secs(1));
            stress_task(1, 1.0);
            std::thread::sleep(Duration::from_secs(1));
            stress_task(1, 1.0);
            std::thread::sleep(Duration::from_secs(1));
            stress_task(1, 1.0);
        });

        for _ in 0..(20000 / MIN_DISCRT_PERIOD_MS) {
            let actual_t = get_temp();

            curr_freq -= freq_ctl.get_delta_freq(actual_t);
            curr_freq = curr_freq.clamp(*MIN_CPU_FREQ, *MAX_CPU_FREQ);

            temperature_velocity_deviation_power += freq_ctl.temp_velocity_err.abs() as i64;

            if freq_ctl.temp_velocity_err.abs() < 0.5 && (actual_t - target_t).abs() < 1000 {
                if optimal_freq.load(Relaxed) == 0 {
                    optimal_freq.store(curr_freq, Relaxed);
                } else {
                    let weight = 2.0 * (0.5 - freq_ctl.temp_velocity_err.abs());
                    let soft_weight = 0.95 * weight;
                    optimal_freq.store(
                        (soft_weight * curr_freq as f64
                            + (1.0 - soft_weight) * optimal_freq.load(Relaxed) as f64)
                            as i32,
                        Relaxed,
                    );
                }
            }

            limit_freq_uniform(curr_freq);

            std::thread::sleep(Duration::from_millis(MIN_DISCRT_PERIOD_MS as u64));
        }

        fast_unlock();

        complex_stress_task.join().expect("what");

        println!("deviation is {}", temperature_velocity_deviation_power);

        println!("optimal_freq: {}", optimal_freq.load(Relaxed));

        temperature_velocity_deviation_power
    };

    fast_unlock();

    let solve = |x_arr: [i64; 3], y_arr: [i64; 3]| -> (bool, i64) {
        use mathru::algebra::linear::matrix::{General, Solve};
        use mathru::{algebra::linear::vector::Vector, matrix};

        let x_matrix = matrix![
            x_arr[0].pow(2) as f64, x_arr[0] as f64, 1.0;
            x_arr[1].pow(2) as f64, x_arr[1] as f64, 1.0;
            x_arr[2].pow(2) as f64, x_arr[2] as f64, 1.0
        ];
        let y_vector = Vector::new_column(y_arr.map(|v| v as f64).to_vec());

        let coeff_vector = x_matrix.solve(&y_vector).unwrap();

        if coeff_vector[0] <= 0.0 {
            println!("a value is {}", coeff_vector[0]);
            return (false, 0);
        } else if coeff_vector[1] > 0.0 {
            println!("b value is {}", coeff_vector[1]);
            return (false, 0);
        } else {
            return (true, (-coeff_vector[1] / (2.0 * coeff_vector[0])) as i64);
        }
    };

    let mut x_mul = [15i64, 50, 150];
    let mut power = [
        test(x_mul[0] as i32),
        test(x_mul[1] as i32),
        test(x_mul[2] as i32),
    ];

    let min_x = x_mul[power
        .iter()
        .position(|x| *x == *power.iter().min().unwrap())
        .unwrap()];

    let lower_x = (min_x / 2).clamp(1, i64::MAX);
    let upper_x = min_x * 3 / 2;

    x_mul = [lower_x, min_x, upper_x];
    power = [
        test(x_mul[0] as i32),
        test(x_mul[1] as i32),
        test(x_mul[2] as i32),
    ];
    let (is_minimum, mut optimal_multiplier_64) = solve(x_mul, power);

    if !is_minimum || optimal_multiplier_64 <= 0 || optimal_multiplier_64 > i32::MAX as i64 {
        optimal_multiplier_64 = x_mul[power
            .iter()
            .position(|x| *x == *power.iter().min().unwrap())
            .unwrap()];
    }

    let optimal_multiplier = optimal_multiplier_64 as i32;

    println!("optimal multiplier is {}", optimal_multiplier);

    let clamp_min_freq = (0.7 * (optimal_freq.load(Relaxed) - *MIN_CPU_FREQ) as f64) as i32;

    let clamp_found: bool;
    if clamp_min_freq <= 0 || clamp_min_freq == *MIN_CPU_FREQ {
        println!("Optimal minimum clamp cpu frequency not found :(");
        clamp_found = false;
    } else {
        println!("Optimal minimum clamp cpu frequency is {}", clamp_min_freq);
        clamp_found = true;
    }

    println!("Do you want to set new default values? [Y/n]: ");
    let input: String = text_io::read!("{}\n");
    if !input.is_empty() && !input.to_ascii_lowercase().starts_with('y') {
        println!("Ok, goodbye :)");
        return;
    }

    std::fs::write(optimal_multiplier_file, optimal_multiplier.to_string()).expect(&format!(
        "Cannot write to {}",
        optimal_multiplier_file.to_string_lossy()
    ));

    if clamp_found {
        std::fs::write(clamp_min_freq_file, clamp_min_freq.to_string()).expect(&format!(
            "Cannot write to {}",
            clamp_min_freq_file.to_string_lossy()
        ));
    }

    println!("Optimal values have been found!");
}
