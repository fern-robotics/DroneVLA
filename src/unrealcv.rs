//! UnrealCV client matching the Python `unrealcv` package framing.
//!
//! Wire format (little endian):
//!   magic u32 = 0x9E2B83C1 | payload_len u32 | payload
//! Request payload: `"<id>:<command>"`. Response payload is text for pose/action
//! commands and raw image bytes for `vget /camera/<id>/lit png`.

use anyhow::{Context, Result};
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

pub const MAGIC: u32 = 0x9E2B_83C1;

#[derive(Debug)]
pub struct UnrealCVClient { stream: TcpStream, next_id: u32 }

impl UnrealCVClient {
    pub fn connect(addr: impl ToSocketAddrs, timeout: Duration) -> Result<Self> {
        let stream = TcpStream::connect(addr)?;
        stream.set_read_timeout(Some(timeout))?;
        stream.set_write_timeout(Some(timeout))?;
        Ok(Self { stream, next_id: 0 })
    }

    /// Send one framed request and return the raw response payload.
    pub fn request(&mut self, cmd: &str) -> Result<Vec<u8>> {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        let payload = format!("{id}:{cmd}").into_bytes();
        self.stream.write_all(&MAGIC.to_le_bytes())?;
        self.stream.write_all(&(payload.len() as u32).to_le_bytes())?;
        self.stream.write_all(&payload)?;
        self.stream.flush()?;

        let mut magic = [0u8; 4];
        let mut size = [0u8; 4];
        self.stream.read_exact(&mut magic)?;
        self.stream.read_exact(&mut size)?;
        let got = u32::from_le_bytes(magic);
        if got != MAGIC { anyhow::bail!("bad UnrealCV magic: 0x{got:08x}"); }
        let n = u32::from_le_bytes(size) as usize;
        let mut body = vec![0u8; n];
        self.stream.read_exact(&mut body)?;
        Ok(body)
    }

    pub fn command(&mut self, cmd: &str) -> Result<String> {
        let body = self.request(cmd)?;
        Ok(String::from_utf8_lossy(&body).trim_end_matches(['\r', '\n']).to_string())
    }

    pub fn get_obj_location(&mut self, name: &str) -> Result<Vec<f64>> {
        parse_vec(&self.command(&format!("vget /object/{name}/location"))?)
    }
    pub fn get_obj_rotation(&mut self, name: &str) -> Result<Vec<f64>> {
        parse_vec(&self.command(&format!("vget /object/{name}/rotation"))?)
    }
    pub fn set_obj_location(&mut self, name: &str, xyz: [f64; 3]) -> Result<()> {
        self.command(&format!("vset /object/{name}/location {} {} {}", xyz[0], xyz[1], xyz[2]))?;
        Ok(())
    }
    pub fn set_rotation(&mut self, name: &str, yaw: f64) -> Result<()> {
        self.command(&format!("vset /object/{name}/rotation 0 0 {yaw}"))?;
        Ok(())
    }
    /// Raw PNG bytes, matching Python `client.get_image(cam, 'lit', mode='png')`.
    pub fn get_image_png(&mut self, camera: u32) -> Result<Vec<u8>> {
        self.request(&format!("vget /camera/{camera}/lit png"))
    }
    pub fn get_image_line(&mut self, camera: u32, mode: &str) -> Result<String> {
        self.command(&format!("vget /camera/{camera}/{mode}"))
    }
}

fn parse_vec(s: &str) -> Result<Vec<f64>> {
    let vals: Vec<f64> = s.split_whitespace().filter_map(|x| x.parse().ok()).collect();
    if vals.is_empty() { anyhow::bail!("not a numeric vector: {s:?}"); }
    Ok(vals)
}

/// Start an Unreal engine and wait for its UnrealCV port.
pub struct EngineSupervisor { child: Child, pub port: u16 }

impl EngineSupervisor {
    pub fn start(program: impl AsRef<Path>, args: &[String], cwd: Option<&Path>, port: u16, wait_s: u64) -> Result<Self> {
        let mut cmd = Command::new(program.as_ref());
        cmd.args(args).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
        if let Some(cwd) = cwd { cmd.current_dir(cwd); }
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        let child = cmd.spawn().with_context(|| format!("spawn engine {}", program.as_ref().display()))?;
        let guard = Self { child, port };
        guard.wait_ready(wait_s)?;
        Ok(guard)
    }
    pub fn wait_ready(&self, wait_s: u64) -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(wait_s);
        while Instant::now() < deadline {
            if TcpStream::connect(("127.0.0.1", self.port)).is_ok() { return Ok(()); }
            std::thread::sleep(Duration::from_millis(200));
        }
        anyhow::bail!("UnrealCV port {} not ready in {}s", self.port, wait_s)
    }
}

impl Drop for EngineSupervisor {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            let _ = Command::new("kill").arg("-9").arg(format!("-{}", self.child.id())).status();
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn frame(payload: &[u8]) -> Vec<u8> {
        let mut out = MAGIC.to_le_bytes().to_vec();
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn parses_vector() { assert_eq!(parse_vec("1.0 2 3").unwrap(), vec![1.0, 2.0, 3.0]); }

    #[test]
    fn framed_roundtrip_text_and_png() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            for i in 0..3 {
                let mut h = [0u8; 8]; s.read_exact(&mut h).unwrap();
                let n = u32::from_le_bytes(h[4..8].try_into().unwrap()) as usize;
                let mut body = vec![0u8; n]; s.read_exact(&mut body).unwrap();
                assert!(body.starts_with(format!("{i}:").as_bytes()));
                if i == 0 { s.write_all(&frame(b"10.0 20.0 30.0\n")).unwrap(); }
                else if i == 1 { s.write_all(&frame(b"ok\n")).unwrap(); }
                else {
                    let png = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
                    s.write_all(&frame(&png)).unwrap();
                }
            }
        });
        let mut c = UnrealCVClient::connect(addr, Duration::from_secs(2)).unwrap();
        assert_eq!(c.get_obj_location("Player").unwrap(), vec![10.0, 20.0, 30.0]);
        c.set_obj_location("Player", [1.0, 2.0, 3.0]).unwrap();
        assert_eq!(c.get_image_png(0).unwrap()[..4], [0x89, b'P', b'N', b'G']);
    }
}
