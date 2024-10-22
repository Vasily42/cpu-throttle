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
extern crate nix;
extern crate num_cpus;
extern crate posixmq;
extern crate serde;
extern crate serde_json;

use clap::{Parser, Subcommand};
use core::{f64, time::Duration};
use serde::{Deserialize, Serialize};
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

const CONFIG_PATH: &str = "/etc/cpu-throttle/config.json";
const DEFAULT_MAX_DESCENT_VELOCITY: f64 = 2.0;
const DEFAULT_MIN_DISCRT_PERIOD_MS: u16 = 150;
const DEFAULT_MAX_DISCRT_PERIOD_MS: u16 = 1500;
const DEFAULT_THROTTLING_START_TIME_MS: u16 = 7000;
const DEFAULT_THROTTLING_RELEASE_TIME_MS: u16 = 12000;
const DEFAULT_CORE_IDLENESS_FACTOR_MS: u16 = 7000;

#[derive(Serialize, Deserialize, Clone, Copy)]
struct JsonConfig {
    min_freq: i32,
    max_descent_velocity: f64,
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
            min_freq: *MIN_CPU_FREQ,
            max_descent_velocity: DEFAULT_MAX_DESCENT_VELOCITY,
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
        let pd = PDController::new(target_t, config);
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
                self.curr_freq -= delta_freq.max(*MAX_STEP_DOWN);
                self.curr_freq = self.curr_freq.clamp(self.config.min_freq, *MAX_CPU_FREQ);

                self.limiter.limit_freq(self.curr_freq);
            } else {
                if self.curr_freq != *MAX_CPU_FREQ {
                    self.curr_freq = *MAX_CPU_FREQ;
                    self.limiter.limit_freq(*MAX_CPU_FREQ);
                }
                self.pd_ctl.prev_t = actual_t;
            }

            new_dscrt_period = self.config.min_period_ms as i32
                + ((self.config.max_period_ms - self.config.min_period_ms) as f64
                    * (1.0 - self.overall_restlessness)) as i32;
        } else {
            if self.curr_freq < *MAX_CPU_FREQ || delta_freq > 0 {
                self.curr_freq -= delta_freq.max(*MAX_STEP_DOWN);
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

static MAX_STEP_DOWN: LazyLock<i32> = LazyLock::new(|| -(*MAX_CPU_FREQ - *MIN_CPU_FREQ) / 10);

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
    max_descent_velocity: f64,
    dynamic_multiplier: f64,
    accel_m: f64,
    decel_m: f64,
}

impl PDController {
    fn new(target_t: i32, config: JsonConfig) -> Self {
        const ACCEL10_TIME_MS: f64 = 800.0;
        const DECEL10_TIME_MS: f64 = 700.0;
        let accel = (10.0_f64).powf(1.0 / (ACCEL10_TIME_MS / config.min_period_ms as f64));
        let decel = 1.0 / (10.0_f64).powf(1.0 / (DECEL10_TIME_MS / config.min_period_ms as f64));
        Self {
            target_t,
            prev_t: get_temp(),
            temp_velocity_err: 0.0,
            max_descent_velocity: config.max_descent_velocity,
            dynamic_multiplier: 1.0,
            accel_m: accel,
            decel_m: decel,
        }
    }

    fn get_delta_freq(&mut self, t: i32) -> i32 {
        let current_t = t;

        let dt = current_t - self.prev_t;
        self.prev_t = current_t;

        let temp_velocity = dt as f64 / DISCRT_PERIOD_MS.load(Relaxed) as f64;

        let proportional_temp_diff = (current_t - self.target_t) as f64 / 1000.0;
        let target_temp_velocity_curve = if proportional_temp_diff > 0.0 {
            self.max_descent_velocity * ((-proportional_temp_diff / self.max_descent_velocity).exp() - 1.0)
        } else {
            -proportional_temp_diff
        };

        let prev_err = self.temp_velocity_err;

        self.temp_velocity_err = temp_velocity - target_temp_velocity_curve;

        if self.temp_velocity_err.abs() > 0.5
            && prev_err.signum() != -self.temp_velocity_err.signum()
        {
            self.dynamic_multiplier *= self.accel_m;
        } else {
            self.dynamic_multiplier *= self.decel_m;
        }
        self.dynamic_multiplier = self.dynamic_multiplier.clamp(1.0, 100.0);

        let grad = self.dynamic_multiplier * 1000.0 * (self.temp_velocity_err);

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
                self.cpu_idleness[i] = self.cpu_idleness[i].min(self.core_idleness_factor_ms);
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
        send_msg(args.command);

        return Ok(());
    } else {
        use InterThreadMessage::*;
        match args.command {
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
                        algo.limiter.limit_freq(*MAX_CPU_FREQ);
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
            }
        }

        if paused {
            continue;
        }

        let new_dscrt_period = algo.step();

        DISCRT_PERIOD_MS.store(new_dscrt_period, Ordering::Relaxed);
    }

    println!("exiting...");

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
