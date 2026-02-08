/*
   Copyright (C) 2024-2026 vmbat2004@gmail.com

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

use clap::{Parser, Subcommand};
use core::{f64, time::Duration};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, Permissions},
    i32,
    io::{Read, Seek, Write},
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::Command,
    str::from_utf8,
    sync::{
        atomic::{
            AtomicBool, AtomicI32,
            Ordering::{self, *},
        },
        Arc, LazyLock,
    },
};

const CONFIG_DIR: &str = "/etc/cpu-throttle";

#[derive(Serialize, Deserialize, Clone, Copy)]
#[serde(default)]
struct JsonConfig {
    min_freq: i32,
    min_multiplier: u16,
    max_multiplier: u16,
    full_throttle_min_time_ms: i32,
    max_descent_velocity: f64,
    min_period_ms: u16,
    max_period_ms: u16,
    start_lag_time_ms: u16,
    release_lag_time_ms: u16,
    core_idleness_factor_ms: u16,
    has_lag: bool,
    multicore_limiter_allowed: bool,
}

impl Default for JsonConfig {
    fn default() -> Self {
        JsonConfig {
            min_freq: *MIN_CPU_FREQ,
            min_multiplier: 5,
            max_multiplier: 150,
            full_throttle_min_time_ms: 4000,
            max_descent_velocity: 2.0,
            min_period_ms: 150,
            max_period_ms: 1500,
            start_lag_time_ms: 7000,
            release_lag_time_ms: 12000,
            core_idleness_factor_ms: 1000,
            has_lag: true,
            multicore_limiter_allowed: true,
        }
    }
}

struct ThrottlingAlgo {
    config: JsonConfig,
    pd_ctl: PDController,
    limiter: Box<dyn FrequencyLimiter>,
    active: bool,
    overall_restlessness: f64,
    curr_freq: i32,
    max_step_down: i32,
}

impl ThrottlingAlgo {
    fn new(target_t: i32, config: JsonConfig) -> Self {
        let pd = PDController::new(target_t, config);
        let mut limiter: Box<dyn FrequencyLimiter> = if config.multicore_limiter_allowed {
            match CPU_PATHS.len() {
                1 => Box::new(UniformFrequencyLimiter::new()),
                2.. => Box::new(MulticoreFrequencyLimiter::new(config.core_idleness_factor_ms)),
                _ => {
                    eprintln!("wtf");
                    panic!()
                }
            }
        } else {
            Box::new(UniformFrequencyLimiter::new())
        };
        limiter.limit_freq(*MAX_CPU_FREQ);
        Self {
            config,
            pd_ctl: pd,
            limiter,
            active: false,
            overall_restlessness: 0.0,
            curr_freq: *MAX_CPU_FREQ,
            max_step_down: (*MAX_CPU_FREQ - *MIN_CPU_FREQ)
                / (config.full_throttle_min_time_ms / config.min_period_ms as i32).max(1),
        }
    }

    fn step(&mut self) -> i32 {
        let actual_t = get_temp();
        let delta_freq = self.pd_ctl.get_delta_freq(actual_t);
        let new_dscrt_period;

        let warmup = delta_freq > 0 && actual_t > self.pd_ctl.target_t - 5000
            || self.curr_freq != *MAX_CPU_FREQ;

        if self.config.has_lag {
            if warmup {
                self.overall_restlessness +=
                    DISCRT_PERIOD_MS.load(Relaxed) as f64 / self.config.start_lag_time_ms as f64;
                self.overall_restlessness = self.overall_restlessness.min(1.0);
            } else if self.curr_freq == *MAX_CPU_FREQ {
                self.overall_restlessness -=
                    DISCRT_PERIOD_MS.load(Relaxed) as f64 / self.config.release_lag_time_ms as f64;
                self.overall_restlessness = self.overall_restlessness.max(0.0);
            }

            if self.overall_restlessness == 1.0 {
                self.active = true;
            }
            if self.overall_restlessness == 0.0 {
                self.active = false;
            }

            if warmup {
                new_dscrt_period = (self.config.min_period_ms as i32
                    + ((self.config.max_period_ms - self.config.min_period_ms) as f64
                        * (1.0 - self.overall_restlessness)) as i32)
                    .min(DISCRT_PERIOD_MS.load(Relaxed));
            } else {
                new_dscrt_period = if self.overall_restlessness == 0.0 {
                    self.config.max_period_ms as i32
                } else {
                    DISCRT_PERIOD_MS.load(Relaxed)
                }
            }
        } else {
            self.active = warmup;
            if warmup {
                new_dscrt_period = self.config.min_period_ms as i32;
            } else {
                new_dscrt_period = self.config.max_period_ms as i32;
            }
        }

        if self.active {
            self.curr_freq -= delta_freq.min(self.max_step_down);
            self.curr_freq = self.curr_freq.clamp(self.config.min_freq, *MAX_CPU_FREQ);
            self.limiter.limit_freq(self.curr_freq);
        }

        new_dscrt_period
    }
}

static CPU_PATHS: LazyLock<Vec<PathBuf>> = LazyLock::new(|| {
    fs::read_dir("/sys/devices/system/cpu")
        .expect("cannot read /sys/devices/system/cpu")
        .flatten()
        .filter_map(|e| e.path().is_dir().then(|| e.path()))
        .filter_map(|path| {
            path.file_name()
                .filter(|f| {
                    let s = f.to_str().unwrap();
                    s.starts_with("cpu") && s.chars().any(|c| c.is_ascii_digit())
                })
                .and(Some(path.to_owned().join("cpufreq")))
        })
        .collect()
});

static MAX_CPU_FREQ: LazyLock<i32> = LazyLock::new(|| {
    read_i32(&CPU_PATHS.get(0).expect("There are no cpus").join("cpuinfo_max_freq"))
});

static MIN_CPU_FREQ: LazyLock<i32> = LazyLock::new(|| {
    read_i32(&CPU_PATHS.get(0).expect("There are no cpus").join("cpuinfo_min_freq"))
});

static TEMPERATURE_PROVIDER_FILE: LazyLock<PathBuf> = LazyLock::new(|| {
    const HWMON_PATH: &str = "/sys/class/hwmon";
    const TARGET_NAMES: &[&str] = &["coretemp", "k10temp", "zenpower", "amd_energy"];
    const TARGET_LABELS: &[&str] = &["Package", "Tdie", "Tctl"];

    let is_target_hwmon = |path: &PathBuf| {
        fs::read_to_string(path.join("name"))
            .ok()
            .map_or(false, |name| TARGET_NAMES.contains(&name.trim()))
    };

    let is_temp_label = |path: &PathBuf| {
        path.file_name()
            .and_then(|n| n.to_str())
            .map_or(false, |s| s.contains("temp") && s.contains("label"))
    };

    let has_target_label = |path: &PathBuf| {
        fs::read_to_string(path).ok().map_or(false, |content| {
            let label = content.trim();
            TARGET_LABELS.iter().any(|&target| label.starts_with(target))
        })
    };

    let to_input_path = |path: PathBuf| {
        path.file_name()
            .and_then(|n| n.to_str())
            .map(|name| path.with_file_name(name.replace("label", "input")))
    };

    fs::read_dir(HWMON_PATH)
        .expect("cannot read /sys/class/hwmon")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(is_target_hwmon)
        .flat_map(|dir| {
            fs::read_dir(&dir)
                .expect("cannot read hwmon directory")
                .filter_map(Result::ok)
                .map(|e| e.path())
        })
        .filter(|p| p.is_file() && is_temp_label(p) && has_target_label(p))
        .find_map(to_input_path)
        .expect("sensor files have not been found")
});

static DISCRT_PERIOD_MS: LazyLock<AtomicI32> =
    LazyLock::new(|| AtomicI32::new(read_config().unwrap_or_default().min_period_ms as i32));

static MQUEUE: LazyLock<posixmq::PosixMq> = LazyLock::new(|| get_mqueue());

struct PDController {
    target_t: i32,
    prev_t: i32,
    temp_velocity_err: f64,
    max_descent_velocity: f64,
    dynamic_multiplier_raw: f64,
    dynamic_multiplier_smoothed: f64,
    smoothing_coeff: f64,
    min_multiplier: f64,
    max_multiplier: f64,
    accel_m: f64,
    decel_m: f64,
}

impl PDController {
    fn new(target_t: i32, config: JsonConfig) -> Self {
        const ACCEL10_TIME_MS: f64 = 300.0;
        let accel = (10.0_f64).powf(1.0 / (ACCEL10_TIME_MS / config.min_period_ms as f64));
        let decel = 1.0 / (10.0_f64).powf(1.0 / (ACCEL10_TIME_MS / config.min_period_ms as f64));
        let smoothing_period = (4.0 * (ACCEL10_TIME_MS / config.min_period_ms as f64)).max(1.0);
        Self {
            target_t,
            prev_t: get_temp(),
            temp_velocity_err: 0.0,
            max_descent_velocity: config.max_descent_velocity,
            dynamic_multiplier_raw: 1.0,
            dynamic_multiplier_smoothed: 1.0,
            smoothing_coeff: 1.0 - 2.0 / (smoothing_period + 1.0),
            min_multiplier: config.min_multiplier as f64,
            max_multiplier: config.max_multiplier as f64,
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
        let target_temp_velocity_curve = (-proportional_temp_diff).max(-self.max_descent_velocity);

        let prev_err = self.temp_velocity_err;
        self.temp_velocity_err = temp_velocity - target_temp_velocity_curve;

        self.dynamic_multiplier_raw *= if self.temp_velocity_err.signum() != prev_err.signum() {
            self.decel_m
        } else {
            self.accel_m
        };

        self.dynamic_multiplier_raw =
            self.dynamic_multiplier_raw.clamp(self.min_multiplier, self.max_multiplier);

        self.dynamic_multiplier_smoothed = self.smoothing_coeff * self.dynamic_multiplier_smoothed
            + (1.0 - self.smoothing_coeff) * self.dynamic_multiplier_raw;

        let grad = self.dynamic_multiplier_smoothed * 1000.0 * (self.temp_velocity_err);

        grad as i32
    }
}

trait FrequencyLimiter {
    fn limit_freq(&mut self, freq: i32);
}

struct UniformFrequencyLimiter {
    freq_ctl_files: Vec<File>,
}

impl UniformFrequencyLimiter {
    fn new() -> Self {
        UniformFrequencyLimiter {
            freq_ctl_files: CPU_PATHS
                .iter()
                .map(|path| {
                    fs::OpenOptions::new()
                        .write(true)
                        .open(path.join("scaling_max_freq"))
                        .expect("cannot open scaling_max_freq")
                })
                .collect(),
        }
    }
}
read_i32
impl FrequencyLimiter for UniformFrequencyLimiter {
    fn limit_freq(&mut self, freq: i32) {
        for file in self.freq_ctl_files.iter_mut() {
            fs::File::seek(file, std::io::SeekFrom::Start(0))
                .expect("cannot seek scaling_max_freq");
            write!(file, "{}", freq).expect("cannot write scaling_max_freq");
        }
    }
}

struct MulticoreFrequencyLimiter {
    freq_ctl_files: Vec<File>,
    freq_check_files: Vec<File>,
    cpu_idleness: Vec<u16>,
    core_idleness_factor_ms: u16,
}

impl MulticoreFrequencyLimiter {
    fn new(core_idleness_factor_ms: u16) -> Self {
        MulticoreFrequencyLimiter {
            freq_ctl_files: CPU_PATHS
                .iter()
                .map(|path| {
                    fs::OpenOptions::new()
                        .write(true)
                        .open(path.join("scaling_max_freq"))
                        .expect("cannot open scaling_max_freq")
                })
                .collect(),
            freq_check_files: CPU_PATHS
                .iter()
                .map(|path| {
                    fs::OpenOptions::new()
                        .read(true)
                        .open(path.join("scaling_cur_freq"))
                        .expect("cannot open scaling_cur_freq")
                })
                .collect(),
            cpu_idleness: vec![core_idleness_factor_ms; CPU_PATHS.len()],
            core_idleness_factor_ms,
        }
    }
}

impl FrequencyLimiter for MulticoreFrequencyLimiter {
    fn limit_freq(&mut self, freq: i32) {
        for ((ctl_file, check_file), idleness) in self
            .freq_ctl_files
            .iter_mut()
            .zip(self.freq_check_files.iter_mut())
            .zip(self.cpu_idleness.iter_mut())
        {
            let curr_freq: i32 = {
                fs::File::seek(check_file, std::io::SeekFrom::Start(0))
                    .expect("cannot seek scaling_cur_freq");
                let mut content = String::new();
                check_file.read_to_string(&mut content).expect("cannot read scaling_cur_freq");
                content.trim().parse().expect("scaling_cur_freq gave not an i32")
            };

            if curr_freq > *MIN_CPU_FREQ + ((freq - *MIN_CPU_FREQ) as f64 * 0.8) as i32 {
                *idleness = 0;
            } else if curr_freq <= *MIN_CPU_FREQ + ((freq - *MIN_CPU_FREQ) as f64 * 0.2) as i32 {
                *idleness += DISCRT_PERIOD_MS.load(Relaxed) as u16;
                *idleness = (*idleness).min(self.core_idleness_factor_ms);
            }

            fs::File::seek(ctl_file, std::io::SeekFrom::Start(0))
                .expect("cannot seek scaling_max_freq");
            write!(
                ctl_file,
                "{}",
                if *idleness >= self.core_idleness_factor_ms { *MAX_CPU_FREQ } else { freq }
            )
            .expect("cannot write scaling_max_freq");
        }
    }
}

#[derive(Serialize, Deserialize, Subcommand, PartialEq, Clone)]
enum InterThreadMessage {
    /// Pause throttling
    Pause,
    /// Continue throttling
    Continue,
    /// Pause/Continue throttling
    Toggle,
    /// Read config.json again and apply it
    ReadConfig,
    /// Exit all threads, finish with success code
    Exit,
    /// Set target temperature
    At(TempArg),
    /// Switch symbolic link config.json to profiles/<config_name>.json and apply it immediately
    SwitchConfig(SwitchConfigArg),
}

#[derive(Serialize, Deserialize, clap::Args, Clone, PartialEq)]
struct TempArg {
    temperature: i32,
}

#[derive(Serialize, Deserialize, clap::Args, Clone, PartialEq)]
struct SwitchConfigArg {
    config_name: String,
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: InterThreadMessage,
}

fn main() -> Result<(), i32> {
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

    if !is_superuser() {
        eprintln!("Run with sudo!");
        return Err(1);
    }

    let mut target_t = target_temperature * 1000;

    if !Path::new(&(CONFIG_DIR.to_owned() + "/profiles/default.json")).exists() {
        std::process::Command::new("mkdir")
            .arg("-p")
            .arg(CONFIG_DIR.to_owned() + "/profiles")
            .status()
            .expect("cannot create config paths");

        std::fs::File::create(CONFIG_DIR.to_owned() + "/profiles/default.json")
            .expect("Cannot create config.json");
        std::fs::set_permissions(
            CONFIG_DIR.to_owned() + "/profiles/default.json",
            Permissions::from_mode(0o644),
        )
        .expect("Cannot set permissions on config.json");

        if !Path::new(&(CONFIG_DIR.to_owned() + "/config.json")).exists() {
            std::process::Command::new("ln")
                .arg("-s")
                .arg("profiles/default.json")
                .arg(CONFIG_DIR.to_owned() + "/config.json")
                .status()
                .unwrap();
        }
    }

    let mut config = match read_config() {
        Ok(config) => config,
        Err(_) => {
            let default_config = JsonConfig::default();
            write_config(default_config).unwrap();
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
                At(temp_arg) => {
                    target_t = temp_arg.temperature * 1000;
                    algo.pd_ctl.target_t = target_t;
                }
                SwitchConfig(sw_arg) => {
                    switch_config(sw_arg.config_name).unwrap();
                    config = read_config().unwrap();
                    algo = ThrottlingAlgo::new(target_t, config);
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
        .max_msg_len(256)
        .create()
        .open("/cpu-throttle")
        .unwrap();

    if is_superuser() {
        std::fs::set_permissions("/dev/mqueue/cpu-throttle", Permissions::from_mode(0o777))
            .expect("Cannot set permissions");
    }
    mq
}

fn send_msg(msg: InterThreadMessage) {
    MQUEUE.send(0, serde_json::to_string(&msg).unwrap().as_bytes()).expect("Cannot send message");
}

fn receive_msg() -> Option<InterThreadMessage> {
    if MQUEUE.attributes().unwrap().current_messages == 0 {
        None
    } else {
        let mut msg_buffer = [0_u8; 256];
        MQUEUE.recv(&mut msg_buffer).unwrap();
        Some(
            serde_json::from_str::<InterThreadMessage>(
                from_utf8(&msg_buffer).unwrap().trim_end_matches('\0'),
            )
            .unwrap(),
        )
    }
}

fn get_temp() -> i32 {
    read_i32(&TEMPERATURE_PROVIDER_FILE)
}

fn read_i32(path: &PathBuf) -> i32 {
    let mut attempts = 0;
    let mut interval = 250;
    let data = loop {
        match std::fs::read_to_string(path) {
            Ok(data) => break data,
            Err(_) => {
                attempts += 1;
                if attempts == 4 {
                    eprintln!("lost access to {}", path.to_str().unwrap());
                    std::process::exit(1);
                }
                std::thread::sleep(Duration::from_millis(interval));
                interval *= 2;
                continue;
            }
        }
    };
    let trimmed = data.trim();
    let value = trimmed.parse().expect("cannot parse file content as i32");
    value
}

fn write_i32(path: &PathBuf, value: i32) {
    let mut attempts = 0;
    let mut interval = 250;
    loop {
        match std::fs::write(path, value.to_string()) {
            Ok(()) => break,
            Err(_) => {
                attempts += 1;
                if attempts == 4 {
                    eprintln!("lost access to {}", path.to_str().unwrap());
                    std::process::exit(1);
                }
                std::thread::sleep(Duration::from_millis(interval));
                interval *= 2;
                continue;
            }
        }
    }
}

fn switch_config(name: String) -> Result<(), ()> {
    if !Path::new(&(CONFIG_DIR.to_owned() + "/profiles/" + &name + ".json")).exists()
        || !is_superuser()
    {
        return Err(());
    }
    std::process::Command::new("ln")
        .arg("-sf")
        .arg(String::from("profiles/") + &name + ".json")
        .arg(CONFIG_DIR.to_owned() + "/config.json")
        .status()
        .unwrap();
    Ok(())
}

fn read_config() -> Result<JsonConfig, Box<dyn std::error::Error>> {
    let bytes_json = std::fs::read(CONFIG_DIR.to_owned() + "/config.json")?;
    let json = serde_json::from_str::<JsonConfig>(from_utf8(&bytes_json)?)?;
    Ok(json)
}

fn write_config(config: JsonConfig) -> Result<(), std::io::Error> {
    let mut config_file =
        std::fs::OpenOptions::new().write(true).open(CONFIG_DIR.to_owned() + "/config.json")?;
    config_file.write(serde_json::to_string_pretty(&config).unwrap().as_bytes())?;
    Ok(())
}

fn is_superuser() -> bool {
    nix::unistd::Uid::effective().is_root()
}

fn already_run() -> bool {
    let mut pgrep = Command::new("pgrep");
    pgrep.arg("cpu-throttle");
    let output = pgrep
        .output()
        .inspect_err(|_| {
            eprintln!("Fatal: pgrep (procps) not installed");
            std::process::exit(1);
        })
        .unwrap()
        .stdout;
    let output_str = from_utf8(&output).unwrap();
    let lines: Vec<&str> = output_str.split('\n').collect();

    lines.len() > 2
}
