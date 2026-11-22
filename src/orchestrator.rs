//! Process supervision for the 5090 experiment loop.
//!
//! The Python trainer/inference server stay Python. This module owns the
//! control plane around them: one-run-at-a-time locks, bounded retries,
//! timeouts, process-group cleanup, and server readiness checks.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSpec {
    pub program: String,
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
    pub log: Option<PathBuf>,
    pub timeout_s: u64,
    pub retries: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunOutcome {
    pub attempts: u32,
    pub success: bool,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub elapsed_s: f64,
}

impl RunSpec {
    pub fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            cwd: None,
            log: None,
            timeout_s: 3600,
            retries: 0,
        }
    }
    pub fn arg(mut self, a: impl Into<String>) -> Self {
        self.args.push(a.into());
        self
    }
    pub fn timeout(mut self, s: u64) -> Self {
        self.timeout_s = s;
        self
    }
    pub fn retries(mut self, n: u32) -> Self {
        self.retries = n;
        self
    }
}

/// A file-backed single-writer lock. Dropping it releases the lock.
pub struct RunLock {
    path: PathBuf,
}

impl RunLock {
    pub fn acquire(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("run lock already held: {}", path.display()))?;
        Ok(Self { path })
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for RunLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn kill_tree(child: &mut Child) {
    #[cfg(unix)]
    {
        let pid = child.id();
        // Negative PID targets the process group created by process_group(0).
        let _ = Command::new("kill")
            .arg("-9")
            .arg(format!("-{pid}"))
            .status();
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn run_once(spec: &RunSpec) -> Result<(bool, Option<i32>, bool)> {
    let mut cmd = Command::new(&spec.program);
    cmd.args(&spec.args).stdin(Stdio::null());
    if let Some(cwd) = &spec.cwd {
        cmd.current_dir(cwd);
    }
    if let Some(log) = &spec.log {
        if let Some(parent) = log.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let f = std::fs::File::create(log)?;
        cmd.stdout(Stdio::from(f.try_clone()?))
            .stderr(Stdio::from(f));
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawn {}", spec.program))?;
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok((status.success(), status.code(), false));
        }
        if start.elapsed() >= Duration::from_secs(spec.timeout_s) {
            kill_tree(&mut child);
            return Ok((false, None, true));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

pub fn run(spec: &RunSpec) -> Result<RunOutcome> {
    let start = Instant::now();
    let mut last = None;
    let mut attempts = 0;
    for attempt in 0..=spec.retries {
        attempts = attempt + 1;
        let (ok, code, timed_out) = run_once(spec)?;
        last = Some((ok, code, timed_out));
        if ok {
            break;
        }
        if attempt < spec.retries {
            std::thread::sleep(Duration::from_secs(2));
        }
    }
    let (ok, code, timed_out) = last.unwrap();
    Ok(RunOutcome {
        attempts,
        success: ok,
        exit_code: code,
        timed_out,
        elapsed_s: start.elapsed().as_secs_f64(),
    })
}

/// Start a long-lived server and wait for its TCP port. The child is killed on drop.
pub struct ServerGuard {
    child: Child,
    pub port: u16,
}

impl ServerGuard {
    pub fn start(
        program: &str,
        args: &[String],
        cwd: Option<&Path>,
        port: u16,
        wait_s: u64,
    ) -> Result<Self> {
        let mut cmd = Command::new(program);
        cmd.args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        if let Some(cwd) = cwd {
            cmd.current_dir(cwd);
        }
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        let child = cmd
            .spawn()
            .with_context(|| format!("spawn server {program}"))?;
        let guard = Self { child, port };
        guard.wait_ready(wait_s)?;
        Ok(guard)
    }
    pub fn wait_ready(&self, wait_s: u64) -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(wait_s);
        while Instant::now() < deadline {
            if TcpStream::connect(("127.0.0.1", self.port)).is_ok() {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        anyhow::bail!("server did not listen on :{} within {}s", self.port, wait_s)
    }
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        kill_tree(&mut self.child);
    }
}

/// Read a command's stdout lines through a pipe (small helper for CLI smoke tests).
pub fn read_lines(mut child: Child) -> Result<Vec<String>> {
    let stdout = child.stdout.take().context("child has no stdout")?;
    let lines = BufReader::new(stdout)
        .lines()
        .collect::<std::io::Result<Vec<_>>>()?;
    let _ = child.wait();
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn timeout_kills_child() {
        let spec = RunSpec::new("sh").arg("-c").arg("sleep 5").timeout(1);
        let out = run(&spec).unwrap();
        assert!(out.timed_out);
        assert!(!out.success);
        assert!(out.elapsed_s < 4.0);
    }
    #[test]
    fn retry_stops_on_success() {
        let spec = RunSpec::new("sh").arg("-c").arg("exit 0").retries(2);
        let out = run(&spec).unwrap();
        assert!(out.success);
        assert_eq!(out.attempts, 1);
    }
    #[test]
    fn lock_is_exclusive() {
        let p = std::env::temp_dir().join(format!("dronevl-lock-{}", std::process::id()));
        let _ = std::fs::remove_file(&p);
        let a = RunLock::acquire(&p).unwrap();
        assert!(RunLock::acquire(&p).is_err());
        drop(a);
        assert!(RunLock::acquire(&p).is_ok());
    }
}
