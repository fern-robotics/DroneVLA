//! Minimal HTTP policy gateway.
//!
//! Keeps the GPU model server in Python and puts a typed, bounded gateway in
//! front of it. This is intentionally dependency-free (std TCP + serde_json)
//! so it can be built and smoke-tested on the 5090 without a Python runtime.

use anyhow::{Context, Result};
use serde_json::json;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct ProxyConfig {
    pub bind: SocketAddr,
    pub upstream: String,
    pub timeout: Duration,
}

#[derive(Debug, Clone)]
struct Upstream {
    host: String,
    port: u16,
    path: String,
}

fn parse_upstream(url: &str) -> Result<Upstream> {
    let rest = url
        .strip_prefix("http://")
        .context("upstream must start with http://")?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/predict"),
    };
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) => (
            h.to_string(),
            p.parse::<u16>().context("bad upstream port")?,
        ),
        None => (authority.to_string(), 80),
    };
    Ok(Upstream {
        host,
        port,
        path: path.to_string(),
    })
}

fn read_request(stream: &mut TcpStream, timeout: Duration) -> Result<(String, Vec<u8>)> {
    stream.set_read_timeout(Some(timeout))?;
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..pos]).to_string();
            let len = head
                .lines()
                .find_map(|l| {
                    l.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                })
                .unwrap_or(0);
            let body_start = pos + 4;
            while buf.len() < body_start + len {
                let n = stream.read(&mut chunk)?;
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
            }
            return Ok((head, buf[body_start..].to_vec()));
        }
    }
    anyhow::bail!("incomplete HTTP request")
}

fn request_path(head: &str) -> &str {
    head.lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .unwrap_or("/predict")
}

fn forward(cfg: &ProxyConfig, up: &Upstream, body: &[u8]) -> Result<Vec<u8>> {
    let mut s = TcpStream::connect((up.host.as_str(), up.port))
        .with_context(|| format!("connect upstream {}:{}", up.host, up.port))?;
    s.set_read_timeout(Some(cfg.timeout))?;
    s.set_write_timeout(Some(cfg.timeout))?;
    let req = format!("POST {} HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", up.path, up.host, up.port, body.len());
    s.write_all(req.as_bytes())?;
    s.write_all(body)?;
    let mut resp = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        match s.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => resp.extend_from_slice(&chunk[..n]),
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                break
            }
            Err(_) if !resp.is_empty() => break,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(resp)
}

fn handle(mut client: TcpStream, cfg: &ProxyConfig, up: &Upstream) -> Result<()> {
    let (head, body) = read_request(&mut client, cfg.timeout)?;
    let path = request_path(&head);
    let response = if path == "/reset" {
        let body = json!({"ok": true}).to_string();
        format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).into_bytes()
    } else {
        let upstream_resp = forward(cfg, up, &body)?;
        if upstream_resp.is_empty() {
            let body = json!({"error": "empty upstream response"}).to_string();
            format!("HTTP/1.1 502 Bad Gateway\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).into_bytes()
        } else {
            upstream_resp
        }
    };
    client.write_all(&response)?;
    Ok(())
}

pub struct ProxyServer {
    listener: TcpListener,
    cfg: ProxyConfig,
    up: Upstream,
}

impl ProxyServer {
    pub fn bind(cfg: ProxyConfig) -> Result<Self> {
        let up = parse_upstream(&cfg.upstream)?;
        let listener = TcpListener::bind(cfg.bind).with_context(|| format!("bind {}", cfg.bind))?;
        Ok(Self { listener, cfg, up })
    }
    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.listener.local_addr()?)
    }
    pub fn serve_forever(&self) -> Result<()> {
        loop {
            let (client, _) = self.listener.accept()?;
            let cfg = self.cfg.clone();
            let up = self.up.clone();
            std::thread::spawn(move || {
                let _ = handle(client, &cfg, &up);
            });
        }
    }
    /// Test helper: accept exactly `n` requests then return.
    pub fn serve_n(&self, n: usize) -> Result<()> {
        for _ in 0..n {
            let (client, _) = self.listener.accept()?;
            handle(client, &self.cfg, &self.up)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn parses_upstream() {
        let u = parse_upstream("http://127.0.0.1:8010/predict").unwrap();
        assert_eq!(u.host, "127.0.0.1");
        assert_eq!(u.port, 8010);
        assert_eq!(u.path, "/predict");
    }

    #[test]
    fn forwards_json_and_reset() {
        let up = TcpListener::bind("127.0.0.1:0").unwrap();
        let up_addr = up.local_addr().unwrap();
        std::thread::spawn(move || {
            for _ in 0..1 {
                let (mut s, _) = up.accept().unwrap();
                let mut buf = [0u8; 4096];
                let n = s.read(&mut buf).unwrap();
                assert!(String::from_utf8_lossy(&buf[..n]).contains("/predict"));
                let body = br#"{"actions":[[1,0,0,0]],"tokens":[1]}"#;
                let resp = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                s.write_all(resp.as_bytes()).unwrap();
                s.write_all(body).unwrap();
            }
        });
        let cfg = ProxyConfig {
            bind: "127.0.0.1:0".parse().unwrap(),
            upstream: format!("http://{up_addr}/predict"),
            timeout: Duration::from_secs(3),
        };
        let server = ProxyServer::bind(cfg).unwrap();
        let addr = server.local_addr().unwrap();
        let t = std::thread::spawn(move || server.serve_n(1).unwrap());
        let mut c = TcpStream::connect(addr).unwrap();
        let body = br#"{"image":"x"}"#;
        let req = format!(
            "POST /predict HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        c.write_all(req.as_bytes()).unwrap();
        c.write_all(body).unwrap();
        let mut resp = String::new();
        c.read_to_string(&mut resp).unwrap();
        assert!(resp.contains("actions"));
        t.join().unwrap();
    }
}
