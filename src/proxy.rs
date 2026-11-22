//! HTTP policy gateway matching `integrations/uavflow_sim/policy_proxy.py`.
//!
//! Input from UAV-Flow-Eval:
//!   {"image": "<base64>", "proprio": [...], "progress": 0.0, "instr": "..."}
//! Upstream OpenVLA server expects:
//!   {"image_png_b64": "<base64 PNG 224x224>", "state": [...],
//!    "progress": 0.0, "instruction": "..."}
//! Output returned to UAV-Flow-Eval:
//!   {"action": [...], "tokens": [...], "logprobs": [...], "done": false}

use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use image::imageops::FilterType;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::Duration;

pub const RESIZE: u32 = 224;

#[derive(Debug, Clone)]
pub struct ProxyConfig { pub bind: SocketAddr, pub upstream: String, pub timeout: Duration }

#[derive(Debug, Clone)]
struct Upstream { host: String, port: u16, path: String }

fn parse_upstream(url: &str) -> Result<Upstream> {
    let rest = url.strip_prefix("http://").context("upstream must start with http://")?;
    let (authority, path) = match rest.find('/') { Some(i) => (&rest[..i], &rest[i..]), None => (rest, "/predict") };
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse::<u16>().context("bad upstream port")?),
        None => (authority.to_string(), 80),
    };
    Ok(Upstream { host, port, path: path.to_string() })
}

fn read_request(stream: &mut TcpStream, timeout: Duration) -> Result<(String, Vec<u8>)> {
    stream.set_read_timeout(Some(timeout))?;
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let n = stream.read(&mut chunk)?;
        if n == 0 { break; }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..pos]).to_string();
            let len = head.lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap_or(0))).unwrap_or(0);
            let body_start = pos + 4;
            while buf.len() < body_start + len {
                let n = stream.read(&mut chunk)?;
                if n == 0 { break; }
                buf.extend_from_slice(&chunk[..n]);
            }
            return Ok((head, buf[body_start..].to_vec()));
        }
    }
    anyhow::bail!("incomplete HTTP request")
}

fn request_path(head: &str) -> &str {
    head.lines().next().and_then(|l| l.split_whitespace().nth(1)).unwrap_or("/predict")
}

fn resized_png_b64(image_b64: &str) -> Result<String> {
    let raw = B64.decode(image_b64.trim()).context("image is not valid base64")?;
    let img = image::load_from_memory(&raw).context("image is not PNG/JPEG")?.to_rgb8();
    // Python proxy leaves an already-224 image byte-identical.
    if img.width() == RESIZE && img.height() == RESIZE { return Ok(image_b64.to_string()); }
    let resized = image::imageops::resize(&img, RESIZE, RESIZE, FilterType::CatmullRom);
    let mut out = Vec::new();
    image::DynamicImage::ImageRgb8(resized)
        .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .context("encode resized PNG")?;
    Ok(B64.encode(out))
}

/// Translate the UAV-Flow payload into the OpenVLA server payload.
pub fn upstream_payload(body: &[u8]) -> Result<Vec<u8>> {
    let v: Value = serde_json::from_slice(body).context("request body is not JSON")?;
    // Already in upstream shape: forward unchanged.
    if v.get("image_png_b64").is_some() && v.get("image").is_none() { return Ok(body.to_vec()); }
    let image = v.get("image").and_then(|x| x.as_str()).context("missing string field `image`")?;
    let out = json!({
        "image_png_b64": resized_png_b64(image)?,
        "state": v.get("proprio").cloned().unwrap_or_else(|| json!([0, 0, 0, 0, 0, 0])),
        "progress": v.get("progress").cloned().unwrap_or_else(|| json!(0.0)),
        "instruction": v.get("instr").cloned().unwrap_or_else(|| json!("")),
    });
    Ok(serde_json::to_vec(&out)?)
}

fn http_body(resp: &[u8]) -> Option<&[u8]> {
    resp.windows(4).position(|w| w == b"\r\n\r\n").map(|p| &resp[p + 4..])
}

/// Translate the OpenVLA response into the UAV-Flow response.
pub fn downstream_response(resp: &[u8]) -> Result<Vec<u8>> {
    let body = http_body(resp).context("upstream response has no HTTP body")?;
    let v: Value = serde_json::from_slice(body).context("upstream body is not JSON")?;
    let out = json!({
        "action": v.get("actions").cloned().unwrap_or_else(|| json!([])),
        "tokens": v.get("tokens").cloned().unwrap_or(Value::Null),
        "logprobs": v.get("logprobs").cloned().unwrap_or(Value::Null),
        "done": false,
    });
    Ok(serde_json::to_vec(&out)?)
}

fn forward(cfg: &ProxyConfig, up: &Upstream, body: &[u8]) -> Result<Vec<u8>> {
    let mut s = TcpStream::connect((up.host.as_str(), up.port)).with_context(|| format!("connect upstream {}:{}", up.host, up.port))?;
    s.set_read_timeout(Some(cfg.timeout))?;
    s.set_write_timeout(Some(cfg.timeout))?;
    let upstream_body = upstream_payload(body)?;
    let req = format!("POST {} HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", up.path, up.host, up.port, upstream_body.len());
    s.write_all(req.as_bytes())?;
    s.write_all(&upstream_body)?;
    let mut resp = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        match s.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => resp.extend_from_slice(&chunk[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock || e.kind() == std::io::ErrorKind::TimedOut => break,
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
        match forward(cfg, up, &body).and_then(|r| downstream_response(&r)) {
            Ok(payload) => format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", payload.len()).into_bytes().into_iter().chain(payload).collect(),
            Err(e) => {
                let payload = json!({"error": e.to_string()}).to_string();
                format!("HTTP/1.1 502 Bad Gateway\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", payload.len(), payload).into_bytes()
            }
        }
    };
    client.write_all(&response)?;
    Ok(())
}

pub struct ProxyServer { listener: TcpListener, cfg: ProxyConfig, up: Upstream }

impl ProxyServer {
    pub fn bind(cfg: ProxyConfig) -> Result<Self> {
        let up = parse_upstream(&cfg.upstream)?;
        let listener = TcpListener::bind(cfg.bind).with_context(|| format!("bind {}", cfg.bind))?;
        Ok(Self { listener, cfg, up })
    }
    pub fn local_addr(&self) -> Result<SocketAddr> { Ok(self.listener.local_addr()?) }
    pub fn serve_forever(&self) -> Result<()> {
        loop {
            let (client, _) = self.listener.accept()?;
            let cfg = self.cfg.clone(); let up = self.up.clone();
            std::thread::spawn(move || { let _ = handle(client, &cfg, &up); });
        }
    }
    pub fn serve_n(&self, n: usize) -> Result<()> {
        for _ in 0..n { let (client, _) = self.listener.accept()?; handle(client, &self.cfg, &self.up)?; }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};
    use std::net::TcpListener;

    fn tiny_png() -> String {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(8, 8, |x, y| Rgb([(x * 30) as u8, (y * 30) as u8, 100]));
        let mut out = Vec::new();
        image::DynamicImage::ImageRgb8(img).write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png).unwrap();
        B64.encode(out)
    }

    #[test]
    fn translates_payload_and_resizes() {
        let body = serde_json::to_vec(&json!({"image": tiny_png(), "proprio": [1,2,3,4,5,6], "progress": 0.25, "instr": "go"})).unwrap();
        let out = upstream_payload(&body).unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["state"], json!([1,2,3,4,5,6]));
        assert_eq!(v["instruction"], "go");
        let raw = B64.decode(v["image_png_b64"].as_str().unwrap()).unwrap();
        let img = image::load_from_memory(&raw).unwrap();
        assert_eq!((img.width(), img.height()), (RESIZE, RESIZE));
    }

    #[test]
    fn translates_response() {
        let body = br#"{"actions":[[1,0,0,0]],"tokens":[9],"logprobs":[-0.5]}"#;
        let resp = [b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n".as_slice(), body].concat();
        let out = downstream_response(&resp).unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["action"], json!([[1,0,0,0]]));
        assert_eq!(v["tokens"], json!([9]));
        assert_eq!(v["done"], false);
    }

    #[test]
    fn forwards_and_transforms_over_http() {
        let up = TcpListener::bind("127.0.0.1:0").unwrap();
        let up_addr = up.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut s, _) = up.accept().unwrap();
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            loop {
                let n = s.read(&mut chunk).unwrap();
                if n == 0 { break; }
                buf.extend_from_slice(&chunk[..n]);
                if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    let head = String::from_utf8_lossy(&buf[..pos]).to_string();
                    let len = head.lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap_or(0))).unwrap_or(0);
                    if buf.len() >= pos + 4 + len { break; }
                }
            }
            let text = String::from_utf8_lossy(&buf).to_string();
            assert!(text.contains("image_png_b64"));
            assert!(text.contains("state"));
            let body = br#"{"actions":[[1,2,3,4]],"tokens":[1],"logprobs":[-0.1]}"#;
            s.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).as_bytes()).unwrap();
            s.write_all(body).unwrap();
        });
        let cfg = ProxyConfig { bind: "127.0.0.1:0".parse().unwrap(), upstream: format!("http://{up_addr}/predict"), timeout: Duration::from_secs(3) };
        let server = ProxyServer::bind(cfg).unwrap(); let addr = server.local_addr().unwrap();
        let t = std::thread::spawn(move || server.serve_n(1).unwrap());
        let mut c = TcpStream::connect(addr).unwrap();
        let body = serde_json::to_vec(&json!({"image": tiny_png(), "proprio": [0,0,0,0,0,0], "progress": 0, "instr": "x"})).unwrap();
        let req = format!("POST /predict HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n", body.len());
        c.write_all(req.as_bytes()).unwrap(); c.write_all(&body).unwrap();
        let mut resp = String::new(); c.read_to_string(&mut resp).unwrap();
        assert!(resp.contains("\"action\":[[1,2,3,4]]"), "{resp}");
        t.join().unwrap();
    }
}
