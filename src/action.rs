//! UAV action representation and the two thesis action interfaces.
//!
//! `delta4` is (dx, dy, dz, dyaw) in metres/radians in the yaw-aligned body
//! frame forward-right-down. The OpenVLA interface normalizes each component
//! with fixed training-quantile bounds and emits four 256-bin tokens per
//! decision. The pi0-FAST interface emits a DCT-compressed token chunk and
//! executes an execution horizon shorter than the predicted chunk.

use crate::geom::{clamp, Vec3};
use serde::{Deserialize, Serialize};
use std::fmt;

/// One four-dimensional UAV motion command.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Delta4 {
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    pub dyaw: f64,
}

impl Delta4 {
    pub const fn zero() -> Self {
        Self {
            dx: 0.0,
            dy: 0.0,
            dz: 0.0,
            dyaw: 0.0,
        }
    }
    pub fn new(dx: f64, dy: f64, dz: f64, dyaw: f64) -> Self {
        Self { dx, dy, dz, dyaw }
    }
    pub fn is_finite(&self) -> bool {
        self.dx.is_finite() && self.dy.is_finite() && self.dz.is_finite() && self.dyaw.is_finite()
    }
    pub fn translation(&self) -> Vec3 {
        Vec3::new(self.dx, self.dy, self.dz)
    }
    pub fn step_norm(&self) -> f64 {
        self.translation().norm()
    }
    pub fn clipped(&self, max_step: f64, max_yaw: f64) -> Self {
        let mut out = *self;
        let n = out.step_norm();
        if n > max_step && n > 1e-12 {
            let s = max_step / n;
            out.dx *= s;
            out.dy *= s;
            out.dz *= s;
        }
        out.dyaw = clamp(out.dyaw, -max_yaw, max_yaw);
        out
    }
}

impl fmt::Display for Delta4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "d4({:+.3}, {:+.3}, {:+.3}, {:+.3})",
            self.dx, self.dy, self.dz, self.dyaw
        )
    }
}

/// Fixed per-dimension ranges, calibrated from training demonstrations. The
/// lateral dimension is near-constant in the demonstrations, so it receives an
/// explicit minimum range rather than a zero-width interval.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ActionBounds {
    pub dx: (f64, f64),
    pub dy: (f64, f64),
    pub dz: (f64, f64),
    pub dyaw: (f64, f64),
}

impl Default for ActionBounds {
    fn default() -> Self {
        Self {
            dx: (-0.40, 0.40),
            dy: (-0.05, 0.05),
            dz: (-0.20, 0.20),
            dyaw: (-0.45, 0.45),
        }
    }
}

impl ActionBounds {
    /// Bounds from 1st/99th percentiles of a demonstration set; degenerate
    /// dimensions are widened explicitly.
    pub fn from_samples(samples: &[Delta4]) -> Self {
        let mut b = ActionBounds::default();
        if samples.is_empty() {
            return b;
        }
        let dims = [
            samples.iter().map(|s| s.dx).collect::<Vec<_>>(),
            samples.iter().map(|s| s.dy).collect::<Vec<_>>(),
            samples.iter().map(|s| s.dz).collect::<Vec<_>>(),
            samples.iter().map(|s| s.dyaw).collect::<Vec<_>>(),
        ];
        let bounds = [
            quantile(&dims[0], 0.01),
            quantile(&dims[1], 0.01),
            quantile(&dims[2], 0.01),
            quantile(&dims[3], 0.01),
        ];
        let highs = [
            quantile(&dims[0], 0.99),
            quantile(&dims[1], 0.99),
            quantile(&dims[2], 0.99),
            quantile(&dims[3], 0.99),
        ];
        let min_width = [0.10, 0.05, 0.08, 0.10];
        let mut out = Vec::new();
        for i in 0..4 {
            let (mut lo, mut hi) = (bounds[i], highs[i]);
            if hi - lo < min_width[i] {
                let mid = 0.5 * (lo + hi);
                lo = mid - min_width[i] * 0.5;
                hi = mid + min_width[i] * 0.5;
            }
            out.push((lo, hi));
        }
        b.dx = out[0];
        b.dy = out[1];
        b.dz = out[2];
        b.dyaw = out[3];
        b
    }
}

fn quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((v.len() - 1) as f64 * q).round() as usize;
    v[idx]
}

fn normalize(x: f64, lo: f64, hi: f64) -> f64 {
    if hi - lo < 1e-12 {
        0.0
    } else {
        clamp(2.0 * (x - lo) / (hi - lo) - 1.0, -1.0, 1.0)
    }
}

fn denormalize(x: f64, lo: f64, hi: f64) -> f64 {
    lo + (x + 1.0) * 0.5 * (hi - lo)
}

pub const OPENVLA_BINS: u32 = 256;
pub const FAST_LEVELS: u32 = 64;
pub const FAST_CHUNK: usize = 4;
pub const FAST_EXEC_HORIZON: usize = 2;

/// A quantized motion plan: the tokens a model would emit plus the decoded
/// commands the controller may execute.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionPlan {
    pub codec: String,
    pub tokens: Vec<u32>,
    pub actions: Vec<Delta4>,
    pub exec_horizon: usize,
    pub predicted_horizon: usize,
    pub max_quantization_error: f64,
}

impl MotionPlan {
    pub fn executed(&self) -> &[Delta4] {
        let n = self.exec_horizon.min(self.actions.len());
        &self.actions[..n]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionCodec {
    /// Four 256-bin tokens per decision, one per delta4 dimension.
    OpenVla { bounds: ActionBounds },
    /// DCT-compressed action chunk with fixed 6-bit coefficient levels.
    Fast {
        bounds: ActionBounds,
        chunk: usize,
        exec_horizon: usize,
    },
}

impl Default for ActionCodec {
    fn default() -> Self {
        Self::OpenVla {
            bounds: ActionBounds::default(),
        }
    }
}

impl ActionCodec {
    pub fn openvla(bounds: ActionBounds) -> Self {
        Self::OpenVla { bounds }
    }
    pub fn fast(bounds: ActionBounds) -> Self {
        Self::Fast {
            bounds,
            chunk: FAST_CHUNK,
            exec_horizon: FAST_EXEC_HORIZON,
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Self::OpenVla { .. } => "openvla-256bin",
            Self::Fast { .. } => "pi0fast-dct-chunk",
        }
    }
    pub fn exec_horizon(&self) -> usize {
        match self {
            Self::OpenVla { .. } => 1,
            Self::Fast { exec_horizon, .. } => *exec_horizon,
        }
    }
    pub fn decision_dt(&self, step_dt: f64) -> f64 {
        self.exec_horizon() as f64 * step_dt
    }

    /// Quantize a proposed chunk through the model-visible token interface and
    /// decode it back into executable commands.
    pub fn quantize(&self, proposed: &[Delta4]) -> MotionPlan {
        match self {
            Self::OpenVla { bounds } => {
                let a = proposed.first().copied().unwrap_or_default();
                let tokens = vec![
                    encode_bin(a.dx, bounds.dx, OPENVLA_BINS),
                    encode_bin(a.dy, bounds.dy, OPENVLA_BINS),
                    encode_bin(a.dz, bounds.dz, OPENVLA_BINS),
                    encode_bin(a.dyaw, bounds.dyaw, OPENVLA_BINS),
                ];
                let decoded = Delta4::new(
                    decode_bin(tokens[0], bounds.dx, OPENVLA_BINS),
                    decode_bin(tokens[1], bounds.dy, OPENVLA_BINS),
                    decode_bin(tokens[2], bounds.dz, OPENVLA_BINS),
                    decode_bin(tokens[3], bounds.dyaw, OPENVLA_BINS),
                );
                MotionPlan {
                    codec: self.name().to_string(),
                    tokens,
                    actions: vec![decoded],
                    exec_horizon: 1,
                    predicted_horizon: 1,
                    max_quantization_error: max_dim_error(a, decoded),
                }
            }
            Self::Fast {
                bounds,
                chunk,
                exec_horizon,
            } => {
                let n = *chunk;
                let mut chunk_in = vec![Delta4::zero(); n];
                for (slot, a) in chunk_in.iter_mut().zip(proposed.iter()) {
                    *slot = *a;
                }
                let dims = [
                    chunk_in.iter().map(|a| a.dx).collect::<Vec<_>>(),
                    chunk_in.iter().map(|a| a.dy).collect::<Vec<_>>(),
                    chunk_in.iter().map(|a| a.dz).collect::<Vec<_>>(),
                    chunk_in.iter().map(|a| a.dyaw).collect::<Vec<_>>(),
                ];
                let bound = [bounds.dx, bounds.dy, bounds.dz, bounds.dyaw];
                let mut tokens = Vec::with_capacity(n * 4);
                let mut decoded = vec![Delta4::zero(); n];
                for d in 0..4 {
                    let norm: Vec<f64> = dims[d]
                        .iter()
                        .map(|v| normalize(*v, bound[d].0, bound[d].1))
                        .collect();
                    let coeffs = dct4(&norm);
                    let mut q = Vec::with_capacity(4);
                    for (k, c) in coeffs.iter().enumerate() {
                        // Orthonormal DCT coefficients of a signal in [-1,1]
                        // are bounded by 2; quantize on that fixed range.
                        let qv = encode_signed(*c, 2.0, FAST_LEVELS);
                        let _ = k;
                        q.push(qv);
                        tokens.push(qv);
                    }
                    let rec = idct4(&q.iter().map(|v| decode_signed(*v, 2.0, FAST_LEVELS)).collect::<Vec<_>>());
                    for (i, v) in rec.iter().enumerate() {
                        let dv = denormalize(*v, bound[d].0, bound[d].1);
                        match d {
                            0 => decoded[i].dx = dv,
                            1 => decoded[i].dy = dv,
                            2 => decoded[i].dz = dv,
                            _ => decoded[i].dyaw = dv,
                        }
                    }
                }
                let max_err = chunk_in
                    .iter()
                    .zip(decoded.iter())
                    .map(|(a, b)| max_dim_error(*a, *b))
                    .fold(0.0_f64, f64::max);
                MotionPlan {
                    codec: self.name().to_string(),
                    tokens,
                    actions: decoded,
                    exec_horizon: (*exec_horizon).min(n),
                    predicted_horizon: n,
                    max_quantization_error: max_err,
                }
            }
        }
    }
}

fn max_dim_error(a: Delta4, b: Delta4) -> f64 {
    (a.dx - b.dx)
        .abs()
        .max((a.dy - b.dy).abs())
        .max((a.dz - b.dz).abs())
        .max((a.dyaw - b.dyaw).abs())
}

fn encode_bin(x: f64, range: (f64, f64), bins: u32) -> u32 {
    let n = normalize(x, range.0, range.1);
    let u = ((n + 1.0) * 0.5 * bins as f64).floor() as i64;
    u.clamp(0, bins as i64 - 1) as u32
}

fn decode_bin(token: u32, range: (f64, f64), bins: u32) -> f64 {
    let n = -1.0 + 2.0 * (token as f64 + 0.5) / bins as f64;
    denormalize(n, range.0, range.1)
}

fn encode_signed(x: f64, scale: f64, levels: u32) -> u32 {
    let n = clamp(x / scale, -1.0, 1.0);
    let u = ((n + 1.0) * 0.5 * levels as f64).floor() as i64;
    u.clamp(0, levels as i64 - 1) as u32
}

fn decode_signed(token: u32, scale: f64, levels: u32) -> f64 {
    let n = -1.0 + 2.0 * (token as f64 + 0.5) / levels as f64;
    n * scale
}

/// Orthonormal DCT-II for a length-4 signal.
pub fn dct4(x: &[f64]) -> [f64; 4] {
    let mut out = [0.0; 4];
    for (k, slot) in out.iter_mut().enumerate() {
        let mut s = 0.0;
        for (n, v) in x.iter().enumerate() {
            s += v * (std::f64::consts::PI * (n as f64 + 0.5) * k as f64 / 4.0).cos();
        }
        let scale = if k == 0 { 0.5 } else { 0.5 * 2.0_f64.sqrt() };
        *slot = s * scale;
    }
    out
}

pub fn idct4(c: &[f64]) -> [f64; 4] {
    let mut out = [0.0; 4];
    for (n, slot) in out.iter_mut().enumerate() {
        let mut s = 0.0;
        for (k, v) in c.iter().enumerate() {
            let scale = if k == 0 { 0.5 } else { 0.5 * 2.0_f64.sqrt() };
            s += scale * v * (std::f64::consts::PI * (n as f64 + 0.5) * k as f64 / 4.0).cos();
        }
        *slot = s;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dct_round_trip() {
        let x = [0.2, -0.4, 0.7, 0.1];
        let c = dct4(&x);
        let y = idct4(&c);
        for i in 0..4 {
            assert!((x[i] - y[i]).abs() < 1e-12, "{} {}", x[i], y[i]);
        }
    }

    #[test]
    fn openvla_quantization_is_bounded() {
        let codec = ActionCodec::openvla(ActionBounds::default());
        let a = Delta4::new(0.3, 0.0, -0.1, 0.2);
        let plan = codec.quantize(&[a]);
        assert_eq!(plan.tokens.len(), 4);
        assert_eq!(plan.exec_horizon, 1);
        assert!(plan.max_quantization_error < 0.01);
        assert!(plan.actions[0].dx > 0.0);
    }

    #[test]
    fn fast_chunk_round_trip() {
        let codec = ActionCodec::fast(ActionBounds::default());
        let chunk = [
            Delta4::new(0.2, 0.0, 0.0, 0.1),
            Delta4::new(0.2, 0.0, 0.0, 0.05),
            Delta4::new(0.2, 0.0, 0.0, 0.0),
            Delta4::new(0.2, 0.0, 0.0, -0.05),
        ];
        let plan = codec.quantize(&chunk);
        assert_eq!(plan.tokens.len(), 16);
        assert_eq!(plan.exec_horizon, 2);
        assert!(plan.max_quantization_error < 0.02);
    }

    #[test]
    fn clipped_respects_limits() {
        let a = Delta4::new(2.0, 0.0, 0.0, 3.0);
        let c = a.clipped(0.5, 0.45);
        assert!((c.step_norm() - 0.5).abs() < 1e-9);
        assert!((c.dyaw - 0.45).abs() < 1e-9);
    }
}
