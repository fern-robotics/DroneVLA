//! UnrealCV text client and Unreal engine supervisor.
//!
//! The Python runner still performs model-side image handling. This module
//! gives Rust a small, testable client for the UnrealCV command socket and a
//! supervisor that guarantees the engine process is reaped when a run ends.

use anyhow::{Context, Result};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct UnrealCVClient {
    stream: TcpStream,
    reader: BufReader<TcpStream>,
}

impl UnrealCVClient {
    pub fn connect(addr: impl ToSocketAddrs, timeout: Duration) -> Result<Self> {
        let stream = TcpStream::connect(addr)?;
        stream.set_read_timeout(Some(timeout))?;
        stream.set_write_timeout(Some(timeout))?;
        let reader = BufReader::new(stream.try_clone()?);
        Ok(Self { stream, reader })
    }
    pub fn command(&mut self, cmd: &str) -> Result<String> {
        self.stream.write_all(cmd.as_bytes())?;
        self.stream.write_all(b"\n")?;
        self.stream.flush()?;
        let mut line = String::new();
        self.reader.read_line(&mut line)?;
        Ok(line.trim_end_matches(['\r', '\n']).to_string())
    }
    pub fn get_obj_location(&mut self, name: &str) -> Result<Vec<f64>> {
        parse_vec(&self.command(&format!("vget /object/{name}/location"))?)
    }
    pub fn get_obj_rotation(&mut self, name: &str) -> Result<Vec<f64>> {
        parse_vec(&self.command(&format!("vget /object/{name}/rotation"))?)
    }
    pub fn set_obj_location(&mut self, name: &str, xyz: [f64; 3]) -> Result<()> {
        self.command(&format!(
            "vset /object/{name}/location {} {} {}",
            xyz[0], xyz[1], xyz[2]
        ))?;
        Ok(())
    }
    pub fn set_rotation(&mut self, name: &str, yaw: f64) -> Result<()> {
        self.command(&format!("vset /object/{name}/rotation 0 0 {yaw}"))?;
        Ok(())
    }
    pub fn get_image_line(&mut self, camera: u32, mode: &str) -> Result<String> {
        self.command(&format!("vget /camera/{camera}/{mode}"))
    }
}

fn parse_vec(s: &str) -> Result<Vec<f64>> {
    let vals: Vec<f64> = s
        .split_whitespace()
        .filter_map(|x| x.parse().ok())
        .collect();
    if vals.is_empty() {
        anyhow::bail!("not a numeric vector: {s:?}");
    }
    Ok(vals)
}

/// Start an Unreal engine and wait for its UnrealCV port.
pub struct EngineSupervisor {
    child: Child,
    pub port: u16,
}

impl EngineSupervisor {
    pub fn start(
        program: impl AsRef<Path>,
        args: &[String],
        cwd: Option<&Path>,
        port: u16,
        wait_s: u64,
    ) -> Result<Self> {
        let mut cmd = Command::new(program.as_ref());
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
            .with_context(|| format!("spawn engine {}", program.as_ref().display()))?;
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
        anyhow::bail!("UnrealCV port {} not ready in {}s", self.port, wait_s)
    }
}

impl Drop for EngineSupervisor {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            let _ = Command::new("kill")
                .arg("-9")
                .arg(format!("-{}", self.child.id()))
                .status();
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    #[test]
    fn parses_vector() {
        assert_eq!(parse_vec("1.0 2 3").unwrap(), vec![1.0, 2.0, 3.0]);
    }
    #[test]
    fn client_roundtrip_against_mock() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut r = BufReader::new(s.try_clone().unwrap());
            for _ in 0..2 {
                let mut line = String::new();
                r.read_line(&mut line).unwrap();
                if line.contains("vget") {
                    s.write_all(b"10.0 20.0 30.0\n").unwrap();
                } else {
                    s.write_all(b"ok\n").unwrap();
                }
            }
        });
        let mut c = UnrealCVClient::connect(addr, Duration::from_secs(2)).unwrap();
        assert_eq!(
            c.get_obj_location("Player").unwrap(),
            vec![10.0, 20.0, 30.0]
        );
        c.set_obj_location("Player", [1.0, 2.0, 3.0]).unwrap();
    }
}
