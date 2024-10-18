/*
   Copyright (C) 2024 vmbat2004@gmail.com

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

extern crate clap;
extern crate ctrlc;
extern crate is_superuser;
extern crate mathru;
extern crate nix;
extern crate num_cpus;
extern crate posixmq;
extern crate serde;
extern crate serde_json;
extern crate text_io;

use clap::{Parser, Subcommand};
use core::{f64, time::Duration};
use serde::{Deserialize, Serialize};
use std::{
    cmp::min,
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
        Arc, LazyLock, Mutex,
    },
};

const CONFIG_PATH: &str = "/etc/cpu-throttle/config.json";
const DEFAULT_MULTIPLIER: u16 = 100;
const DEFAULT_MIN_DISCRT_PERIOD_MS: u16 = 150;
const DEFAULT_MAX_DISCRT_PERIOD_MS: u16 = 1500;
const DEFAULT_THROTTLING_START_TIME_MS: u16 = 7000;
const DEFAULT_THROTTLING_RELEASE_TIME_MS: u16 = 12000;
const DEFAULT_CORE_IDLENESS_FACTOR_MS: u16 = 7000;

#[derive(Serialize, Deserialize, Clone, Copy)]
struct JsonConfig {
    multiplier: u16,
    min_freq: i32,
    min_period_ms: u16,
    max_period_ms: u16,
    start_time_ms: u16,
    release_time_ms: u16,
    core_idleness_factor_ms: u16,
    has_idle: bool,
    multicore_limiter_allowed: bool,
}

impl Default for JsonConfig {
    fn default() -> Self {
        JsonConfig {
            multiplier: DEFAULT_MULTIPLIER,
            min_freq: *MIN_CPU_FREQ,
            min_period_ms: DEFAULT_MIN_DISCRT_PERIOD_MS,
            max_period_ms: DEFAULT_MAX_DISCRT_PERIOD_MS,
            start_time_ms: DEFAULT_THROTTLING_START_TIME_MS,
            release_time_ms: DEFAULT_THROTTLING_RELEASE_TIME_MS,
            core_idleness_factor_ms: DEFAULT_CORE_IDLENESS_FACTOR_MS,
            has_idle: true,
            multicore_limiter_allowed: true,
        }
    }
}

struct ThrottlingAlgo {
    config: JsonConfig,
    pd_ctl: PDController,
    limiter: Box<dyn FrequencyLimiter>,
    overall_restlessness: f64,
    curr_freq: i32,
}

impl ThrottlingAlgo {
    fn new(target_t: i32, config: JsonConfig) -> Self {
        let pd = PDController::new(target_t, config.multiplier as i32);
        let limiter: Box<dyn FrequencyLimiter> = if config.multicore_limiter_allowed {
            match *N_CPUS {
                1 => Box::new(UniformFrequencyLimiter),
                2.. => Box::new(MulticoreFrequencyLimiter::new(config.core_idleness_factor_ms)),
                _ => {
                    eprintln!("wtf");
                    panic!()
                }
            }
        } else {
            Box::new(UniformFrequencyLimiter)
        };
        Self { config, pd_ctl: pd, limiter, overall_restlessness: 0.0, curr_freq: *MAX_CPU_FREQ }
    }

    fn step(&mut self) -> i32 {
        let actual_t = get_temp();
        let delta_freq = self.pd_ctl.get_delta_freq(actual_t);
        let new_dscrt_period;

        if self.config.has_idle {
            if delta_freq > 0 && actual_t > self.pd_ctl.target_t - 5000
                || self.curr_freq != *MAX_CPU_FREQ
            {
                self.overall_restlessness +=
                    DISCRT_PERIOD_MS.load(Relaxed) as f64 / self.config.start_time_ms as f64;
                if self.overall_restlessness > 1.0 {
                    self.overall_restlessness = 1.0;
                }
            } else if self.curr_freq == *MAX_CPU_FREQ {
                self.overall_restlessness -=
                    DISCRT_PERIOD_MS.load(Relaxed) as f64 / self.config.release_time_ms as f64;
                if self.overall_restlessness < 0.0 {
                    self.overall_restlessness = 0.0;
                }
            }

            if self.overall_restlessness >= 0.9 {
                self.curr_freq -= delta_freq;
                self.curr_freq = self.curr_freq.clamp(self.config.min_freq, *MAX_CPU_FREQ);

                self.limiter.limit_freq(self.curr_freq);
            } else {
                if self.curr_freq != *MAX_CPU_FREQ {
                    self.curr_freq = *MAX_CPU_FREQ;
                    fast_unlock();
                }
                self.pd_ctl.prev_t = actual_t;
            }

            new_dscrt_period = self.config.min_period_ms as i32
                + ((self.config.max_period_ms - self.config.min_period_ms) as f64
                    * (1.0 - self.overall_restlessness)) as i32;
        } else {
            if self.curr_freq < *MAX_CPU_FREQ || delta_freq > 0 {
                self.curr_freq -= delta_freq;
                self.curr_freq = self.curr_freq.clamp(self.config.min_freq, *MAX_CPU_FREQ);

                self.limiter.limit_freq(self.curr_freq);
            }
            new_dscrt_period = self.config.min_period_ms as i32;
        }

        new_dscrt_period
    }
}

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

    temper_file_list.last().expect("There are no hwmon files").0.to_string()
});

static DISCRT_PERIOD_MS: LazyLock<AtomicI32> =
    LazyLock::new(|| AtomicI32::new(read_config().unwrap_or_default().min_period_ms as i32));

static MQUEUE: LazyLock<posixmq::PosixMq> = LazyLock::new(|| get_mqueue());

struct PDController {
    target_t: i32,
    prev_t: i32,
    temp_velocity_err: f64,
    multiplier: i32,
}

impl PDController {
    fn new(target_t: i32, multiplier: i32) -> Self {
        Self { target_t, prev_t: get_temp(), temp_velocity_err: 0.0, multiplier }
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

        let grad = self.multiplier as f64 * 1000.0 * self.temp_velocity_err;

        grad as i32
    }
}

trait FrequencyLimiter {
    fn limit_freq(&mut self, freq: i32);
}

struct UniformFrequencyLimiter;

impl FrequencyLimiter for UniformFrequencyLimiter {
    fn limit_freq(&mut self, freq: i32) {
        for i in 0..*N_CPUS {
            std::fs::write(
                format!("/sys/devices/system/cpu/cpu{}/cpufreq/scaling_max_freq", i),
                freq.to_string(),
            )
            .expect("Cannot write to /sys");
        }
    }
}

struct MulticoreFrequencyLimiter {
    cpu_idleness: Vec<u16>,
    core_idleness_factor_ms: u16,
}

impl MulticoreFrequencyLimiter {
    fn new(core_idleness_factor_ms: u16) -> Self {
        MulticoreFrequencyLimiter {
            cpu_idleness: vec![core_idleness_factor_ms; *N_CPUS as usize],
            core_idleness_factor_ms,
        }
    }
}

impl FrequencyLimiter for MulticoreFrequencyLimiter {
    fn limit_freq(&mut self, freq: i32) {
        for i in 0..(*N_CPUS as usize) {
            let curr_freq =
                read_i32(&format!("/sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq", i));

            if curr_freq > *MIN_CPU_FREQ + ((freq - *MIN_CPU_FREQ) as f64 * 0.8) as i32 {
                self.cpu_idleness[i] = 0
            } else if curr_freq < *MIN_CPU_FREQ + ((freq - *MIN_CPU_FREQ) as f64 * 0.2) as i32 {
                self.cpu_idleness[i] += DISCRT_PERIOD_MS.load(Relaxed) as u16;
                self.cpu_idleness[i] = self.cpu_idleness[i].clamp(0, self.core_idleness_factor_ms);
            }

            if self.cpu_idleness[i] >= self.core_idleness_factor_ms {
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
    if Path::new("/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies")
        .try_exists()
        .unwrap()
    {
        eprintln!("The CPU probably doesn't support fine-grained frequency scaling");
        return Err(1);
    }

    let args = Args::parse();

    let target_temperature: i32 = if already_run() {
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
    } else {
        use InterThreadMessage::*;
        match args.command {
            InterThreadMessage::Optimize => {
                if !is_superuser::is_superuser() {
                    eprintln!("Run this command [optimize] with sudo!");
                    return Err(1);
                }
                optimize();
                return Ok(());
            }
            At(arg) => arg.temperature,
            _ => {
                eprintln!(
                    "Daemon has not been started. Start it with \'cpu-throttle at <temperature>\'"
                );
                return Err(1);
            }
        }
    };

    let target_t = target_temperature * 1000;

    let mut config = match read_config() {
        Ok(config) => config,
        Err(_) => {
            let default_config = JsonConfig::default();
            write_config(default_config).expect("Cannot write config");
            default_config
        }
    };

    fast_unlock();

    let throttling = Arc::new(AtomicBool::new(true));
    let t_wait_term = throttling.clone();

    ctrlc::set_handler(move || {
        t_wait_term.store(false, Release);
    })
    .expect("Error setting SIGTERM handler");

    let mut algo = ThrottlingAlgo::new(target_t, config);

    let mut paused = false;

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
                        algo.curr_freq = *MAX_CPU_FREQ;
                        algo.overall_restlessness = 0.0;
                        DISCRT_PERIOD_MS.store(config.max_period_ms as i32, Ordering::Relaxed);
                        paused = true;
                    }
                }
                ReadConfig => {
                    config = read_config().unwrap();
                    algo = ThrottlingAlgo::new(target_t, config);
                }
                At(temperature) => {
                    algo.pd_ctl.target_t = temperature.temperature * 1000;
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

        let new_dscrt_period = algo.step();

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
    MQUEUE.send(0, msg.to_string().as_bytes()).expect("Cannot send message");
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

fn fast_unlock() {
    let mut ctl = UniformFrequencyLimiter;
    ctl.limit_freq(*MAX_CPU_FREQ);
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

fn read_config() -> Result<JsonConfig, std::io::Error> {
    let bytes_json = std::fs::read(CONFIG_PATH)?;
    Ok(serde_json::from_str(from_utf8(&bytes_json).unwrap()).expect("json parse error"))
}

fn write_config(config: JsonConfig) -> Result<(), ()> {
    let config_path = Path::new(CONFIG_PATH);
    if !config_path.exists() {
        if !is_superuser::is_superuser() {
            return Err(());
        }
        let parent_dir = config_path.parent().unwrap();
        if !parent_dir.exists() {
            std::fs::DirBuilder::new()
                .create(parent_dir)
                .expect("failed creating parent dir in /etc");
        }
        std::fs::File::create(config_path).expect("Cannot create config.json");
        std::fs::set_permissions(config_path, Permissions::from_mode(0o666))
            .expect("Cannot set permissions on config.json");
    }
    std::fs::write(config_path, serde_json::to_string_pretty(&config).unwrap()).unwrap();
    Ok(())
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

    let config = match read_config() {
        Ok(config) => {
            if config.multiplier != DEFAULT_MULTIPLIER || config.min_freq != *MIN_CPU_FREQ {
                print!("Optimal values were previosly established. Continue anyway? [Y/n]: ");
                let input: String = text_io::read!("{}\n");
                if !input.is_empty() && !input.to_ascii_lowercase().starts_with('y') {
                    println!("Ok, goodbye ;)");
                    return;
                }
            }
            config
        }
        Err(_) => panic!(),
    };

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

    let optimal_freqs: Arc<Mutex<Vec<i32>>> = Arc::new(Mutex::new(Vec::new()));

    let test = |multiplier: i32| -> i64 {
        let stress_task = |sec: i32, load: f64| {
            let ld_cpus = ((*N_CPUS as f64 * load) as i32).clamp(1, *N_CPUS);
            let mut command = Command::new("timeout");
            command.args([&format!("{}s", sec), "stress", "-c", &ld_cpus.to_string()]);
            command.status().expect("Cannot execute stress command");
        };
        println!("Warming up...");
        stress_task(15, 1.0);

        let mut algo = ThrottlingAlgo::new(
            target_t,
            JsonConfig {
                has_idle: false,
                multicore_limiter_allowed: false,
                min_freq: *MIN_CPU_FREQ,
                ..config
            },
        );

        algo.pd_ctl.multiplier = multiplier;

        let mut temperature_velocity_deviation_power = 0i64;

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

        for _ in 0..(20000 / config.min_period_ms) {
            if algo.pd_ctl.temp_velocity_err.abs() < 0.5 && (get_temp() - target_t).abs() < 1000 {
                optimal_freqs.lock().unwrap().push(algo.curr_freq);
            }

            algo.step();

            temperature_velocity_deviation_power += algo.pd_ctl.temp_velocity_err.abs() as i64;

            std::thread::sleep(Duration::from_millis(config.min_period_ms as u64));
        }

        fast_unlock();

        complex_stress_task.join().expect("what");

        println!("deviation is {}", temperature_velocity_deviation_power);

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
    let mut power = [test(x_mul[0] as i32), test(x_mul[1] as i32), test(x_mul[2] as i32)];

    let min_x = x_mul[power.iter().position(|x| *x == *power.iter().min().unwrap()).unwrap()];

    let lower_x = (min_x / 2).clamp(1, i64::MAX);
    let upper_x = min_x * 3 / 2;

    x_mul = [lower_x, min_x, upper_x];
    power = [test(x_mul[0] as i32), test(x_mul[1] as i32), test(x_mul[2] as i32)];
    let (is_minimum, mut optimal_multiplier_64) = solve(x_mul, power);

    if !is_minimum || optimal_multiplier_64 <= 0 || optimal_multiplier_64 > i32::MAX as i64 {
        optimal_multiplier_64 =
            x_mul[power.iter().position(|x| *x == *power.iter().min().unwrap()).unwrap()];
    }

    let optimal_multiplier = optimal_multiplier_64 as i32;

    println!("optimal multiplier is {}", optimal_multiplier);

    let mut clamp_min_freq = *MIN_CPU_FREQ;

    let opt_freq_guard = optimal_freqs.lock().unwrap();
    let len = opt_freq_guard.len();

    if len == 0 {
        println!("Optimal minimum clamp cpu frequency not found :(");
    } else {
        if len % 2 == 1 {
            clamp_min_freq = opt_freq_guard[len / 2];
        } else {
            clamp_min_freq = (opt_freq_guard[len / 2] + opt_freq_guard[len / 2 - 1]) / 2;
        }
        clamp_min_freq = min(clamp_min_freq, opt_freq_guard.iter().sum::<i32>() / len as i32);
        clamp_min_freq = *MIN_CPU_FREQ + (clamp_min_freq - *MIN_CPU_FREQ) / 2;
        println!("Optimal minimum clamp cpu frequency is {}", clamp_min_freq);
    }

    println!("Do you want to set new default values? [Y/n]: ");
    let input: String = text_io::read!("{}\n");
    if !input.is_empty() && !input.to_ascii_lowercase().starts_with('y') {
        println!("Ok, goodbye :)");
        return;
    }

    let new_config =
        JsonConfig { multiplier: optimal_multiplier as u16, min_freq: clamp_min_freq, ..config };

    write_config(new_config).expect("idk");

    println!("Optimal values have been found!");
}
