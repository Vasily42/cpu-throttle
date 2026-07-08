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

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use core::time::Duration;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io,
    io::{Read, Seek, Write},
    path::{Path, PathBuf},
    process,
    str::from_utf8,
    sync::{
        atomic::{
            AtomicBool, AtomicI32,
            Ordering::{Acquire, Relaxed, Release},
        },
        Arc, LazyLock,
    },
    thread,
};

static CPU_PATHS: LazyLock<Vec<PathBuf>> = LazyLock::new(|| {
    let paths: Vec<PathBuf> = fs::read_dir("/sys/devices/system/cpu")
        .expect("cannot read /sys/devices/system/cpu")
        .flatten()
        .filter_map(|e| e.path().is_dir().then_some(e.path()))
        .filter_map(|path| {
            path.file_name()
                .filter(|f| {
                    let s = f.to_str().unwrap();
                    s.starts_with("cpu") && s.chars().any(|c| c.is_ascii_digit())
                })
                .and(Some(path.join("cpufreq")))
        })
        .collect();
    if paths.is_empty() {
        panic!("Cpu paths not found")
    }
    paths
});

static MAX_CPU_FREQ: LazyLock<i32> =
    LazyLock::new(|| read_i32(&CPU_PATHS.first().unwrap().join("cpuinfo_max_freq")));

static MIN_CPU_FREQ: LazyLock<i32> =
    LazyLock::new(|| read_i32(&CPU_PATHS.first().unwrap().join("cpuinfo_min_freq")));

static TEMPERATURE_PROVIDER_FILE: LazyLock<PathBuf> = LazyLock::new(|| {
    const HWMON_PATH: &str = "/sys/class/hwmon";
    const TARGET_NAMES: &[&str] = &["coretemp", "k10temp", "zenpower", "amd_energy"];
    const TARGET_LABELS: &[&str] = &["Package", "Tdie", "Tctl"];

    let is_target_hwmon = |path: &PathBuf| {
        fs::read_to_string(path.join("name")).is_ok_and(|name| TARGET_NAMES.contains(&name.trim()))
    };

    let is_temp_label = |path: &PathBuf| {
        path.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|s| s.contains("temp") && s.contains("label"))
    };

    let has_target_label = |path: &PathBuf| {
        fs::read_to_string(path).ok().is_some_and(|content| {
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
        .flatten()
        .map(|e| e.path())
        .filter(is_target_hwmon)
        .flat_map(|dir| {
            fs::read_dir(&dir).expect("cannot read a hwmon directory").flatten().map(|e| e.path())
        })
        .filter(|p| p.is_file() && is_temp_label(p) && has_target_label(p))
        .find_map(to_input_path)
        .expect("sensor files have not been found")
});

static DISCRT_PERIOD_MS: LazyLock<AtomicI32> =
    LazyLock::new(|| AtomicI32::new(read_config().unwrap_or_default().min_period_ms as i32));

static MQUEUE: LazyLock<posixmq::PosixMq> = LazyLock::new(|| {
    let mq = posixmq::OpenOptions::readwrite()
        .mode(0o777)
        .nonblocking()
        .capacity(3)
        .max_msg_len(256)
        .create()
        .open("/cpu-throttle")
        .expect("Cannot open mqueue");

    if is_superuser() {
        fs::set_permissions(
            "/dev/mqueue/cpu-throttle",
            std::os::unix::fs::PermissionsExt::from_mode(0o777),
        )
        .expect("Cannot set permissions");
    }

    mq
});

static LOCK_FILE: LazyLock<File> = LazyLock::new(|| {
    let lock_file = File::create("/run/cpu-throttle.lock").expect("cannot open lock file");
    let _ = File::set_permissions(&lock_file, std::os::unix::fs::PermissionsExt::from_mode(0o777));
    lock_file
});

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
            prev_t: 0,
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

impl FrequencyLimiter for UniformFrequencyLimiter {
    fn limit_freq(&mut self, freq: i32) {
        for file in self.freq_ctl_files.iter_mut() {
            File::seek(file, io::SeekFrom::Start(0)).expect("cannot seek scaling_max_freq");
            file.write_all(freq.to_string().as_bytes()).expect("cannot write scaling_max_freq");
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
                File::seek(check_file, io::SeekFrom::Start(0))
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

            File::seek(ctl_file, io::SeekFrom::Start(0)).expect("cannot seek scaling_max_freq");
            let freq_to_wr =
                if *idleness >= self.core_idleness_factor_ms { *MAX_CPU_FREQ } else { freq };
            ctl_file
                .write_all(freq_to_wr.to_string().as_bytes())
                .expect("cannot write scaling_max_freq");
        }
    }
}

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
            max_descent_velocity: 0.8,
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
        let mut limiter: Box<dyn FrequencyLimiter> =
            if config.multicore_limiter_allowed && CPU_PATHS.len() > 1 {
                Box::new(MulticoreFrequencyLimiter::new(config.core_idleness_factor_ms))
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
        let actual_t = read_i32(&TEMPERATURE_PROVIDER_FILE);
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

    fn force_unlock_freq(&mut self) {
        self.limiter.limit_freq(*MAX_CPU_FREQ);
        self.curr_freq = *MAX_CPU_FREQ;
        self.overall_restlessness = 0.0;
    }

    fn set_target(&mut self, target_t: i32) {
        self.pd_ctl.target_t = target_t;
    }
}

#[derive(Serialize, Deserialize, Subcommand, PartialEq, Clone)]
enum ControlCommand {
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
#[command(version, about)]
struct Args {
    #[command(subcommand)]
    command: ControlCommand,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if already_run() {
        send_msg(args.command).context("Cannot send message to daemon")?;
        return Ok(());
    }

    let target_temperature: i32;
    match args.command {
        ControlCommand::At(arg) => target_temperature = arg.temperature,
        ControlCommand::SwitchConfig(sw_arg) => {
            switch_config(sw_arg.config_name)
                .context("Cannot switch config, try start the daemon first")?;
            return Ok(());
        }
        ControlCommand::Exit => return Ok(()),
        _ => {
            bail!("Daemon has not been started. Start it with \'cpu-throttle at <temperature>\'");
        }
    }

    if !is_superuser() {
        bail!("Run with sudo!");
    }

    if !Path::new("/etc/cpu-throttle/profiles/default.json").exists() {
        fs::create_dir_all("/etc/cpu-throttle/profiles").context("create_dir_all at main")?;
        File::create("/etc/cpu-throttle/profiles/default.json")
            .context("default config creation at main")?;

        fs::set_permissions(
            "/etc/cpu-throttle/profiles/default.json",
            std::os::unix::fs::PermissionsExt::from_mode(0o644),
        )
        .context("setting permissions at main")?;

        if !Path::new("/etc/cpu-throttle/config.json").exists() {
            std::os::unix::fs::symlink("profiles/default.json", "/etc/cpu-throttle/config.json")
                .context("symlink creation at main")?;
        }

        let default_config = JsonConfig::default();
        write_config(&default_config).context("writing default config at main")?;
    }

    let mut config = read_config().context("reading config at main")?;

    let throttling = Arc::new(AtomicBool::new(true));
    let t_wait_term = Arc::clone(&throttling);

    ctrlc::set_handler(move || {
        t_wait_term.store(false, Release);
    })
    .context("Cannot set ctrl+c handler")?;

    let mut target_t = target_temperature * 1000;
    let mut algo = ThrottlingAlgo::new(target_t, config);

    let mut paused = false;

    while throttling.load(Acquire) {
        thread::sleep(Duration::from_millis(DISCRT_PERIOD_MS.load(Relaxed) as u64));

        let msg = receive_msg();
        if let Ok(msg) = msg {
            match msg {
                ControlCommand::Pause | ControlCommand::Continue | ControlCommand::Toggle => {
                    let paused_n = match msg {
                        ControlCommand::Toggle => !paused,
                        ControlCommand::Continue => false,
                        ControlCommand::Pause => true,
                        _ => panic!("impossible"),
                    };

                    if !paused && paused_n {
                        algo.force_unlock_freq();
                        DISCRT_PERIOD_MS.store(config.max_period_ms as i32, Relaxed);
                    }
                    paused = paused_n;
                }
                ControlCommand::ReadConfig => {
                    config =
                        read_config().context("caught read_config: cannot read second time")?;
                    algo = ThrottlingAlgo::new(target_t, config);
                }
                ControlCommand::At(temp_arg) => {
                    target_t = temp_arg.temperature * 1000;
                    algo.set_target(target_t);
                }
                ControlCommand::SwitchConfig(sw_arg) => {
                    switch_config(sw_arg.config_name)
                        .context("caught switch_config: cannot switch config")?;
                    config = read_config()
                        .context("caught switch_config: cannot read config via config.json")?;
                    algo = ThrottlingAlgo::new(target_t, config);
                }
                ControlCommand::Exit => {
                    break;
                }
            }
        }

        if paused {
            continue;
        }

        let new_dscrt_period = algo.step();

        DISCRT_PERIOD_MS.store(new_dscrt_period, Relaxed);
    }

    println!("exiting...");

    algo.force_unlock_freq();

    posixmq::remove_queue("/cpu-throttle").context("cannot remove posixmq queue at main")?;
    Ok(())
}

fn send_msg(msg: ControlCommand) -> Result<()> {
    MQUEUE.send(0, serde_json::to_string(&msg)?.as_bytes())?;
    Ok(())
}

fn receive_msg() -> Result<ControlCommand> {
    if MQUEUE.attributes().unwrap().current_messages == 0 {
        bail!("No messages")
    } else {
        let mut msg_buffer = [0_u8; 256];
        MQUEUE.recv(&mut msg_buffer)?;
        Ok(serde_json::from_str::<ControlCommand>(
            from_utf8(&msg_buffer)
                .context("invalid utf8 in mqueue message")?
                .trim_end_matches('\0'),
        )
        .context("invalid json in mqueue message")?)
    }
}

fn read_i32(path: &Path) -> i32 {
    let mut attempts = 0;
    let mut interval = 250;
    let data = loop {
        match fs::read_to_string(path) {
            Ok(data) => break data,
            Err(_) => {
                attempts += 1;
                if attempts == 4 {
                    eprintln!("lost access to {}", path.display());
                    process::exit(1);
                }
                thread::sleep(Duration::from_millis(interval));
                interval *= 2;
                continue;
            }
        }
    };
    let trimmed = data.trim();
    trimmed.parse().expect("cannot parse file content as i32")
}

fn switch_config(name: String) -> Result<()> {
    if !Path::new(&format!("/etc/cpu-throttle/profiles/{}.json", name)).exists() || !is_superuser()
    {
        bail!("Need sudo!");
    }
    fs::remove_file("/etc/cpu-throttle/config.json")
        .context("cannot remove config.json link in switch_config")?;
    std::os::unix::fs::symlink(format!("profiles/{}.json", name), "/etc/cpu-throttle/config.json")
        .context("cannot create config.json link in switch_config")?;
    Ok(())
}

fn read_config() -> Result<JsonConfig> {
    let json_string = fs::read_to_string("/etc/cpu-throttle/config.json")
        .context("cannot read config via config.json link")?;
    let json = serde_json::from_str::<JsonConfig>(&json_string)
        .context("invalid json at config.json link")?;
    Ok(json)
}

fn write_config(config: &JsonConfig) -> Result<()> {
    let mut config_file = fs::OpenOptions::new()
        .write(true)
        .open("/etc/cpu-throttle/config.json")
        .context("cannot open config.json with write permissions")?;
    config_file
        .write_all(serde_json::to_string_pretty(config).expect("not possible?").as_bytes())
        .context("cannot write config in opened file")?;
    Ok(())
}

fn is_superuser() -> bool {
    nix::unistd::Uid::effective().is_root()
}

fn already_run() -> bool {
    File::try_lock(&LOCK_FILE).is_err()
}
