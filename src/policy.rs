//! Goal-conditioned linear motion policy. It stands in for the VLA: the fixed
//! intent planner supplies a goal pose, and the learned policy emits delta4
//! motion tokens. SFT uses the actual UAV-Flow-Sim trajectories; GRPO uses
//! AirSimNH rollout rewards.

use crate::action::Delta4;
use crate::geom::{ned_to_body, wrap_angle, Vec3};
use crate::planner::Goal;
use crate::sim::SimState;
use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub const N_FEATURES: usize = 48;
pub const TEXT_BUCKETS: usize = 16;

#[derive(Debug, Clone)]
pub struct FeatureFrame<'a> {
    pub instruction: &'a str,
    pub intent_type: usize,
    pub magnitude: f64,
    pub side_sign: f64,
    pub progress: f64,
    pub completed_fraction: f64,
    pub time_fraction: f64,
    pub state: SimState,
    pub goal: Goal,
    pub landmark: Option<(f64, f64)>, // bearing, range
    pub prev_action: Delta4,
}

pub fn text_hash(instruction: &str) -> [f64; TEXT_BUCKETS] {
    let mut counts = [0.0f64; TEXT_BUCKETS];
    let mut n = 0.0;
    for token in instruction
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
    {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in token.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        counts[(h as usize) % TEXT_BUCKETS] += 1.0;
        n += 1.0;
    }
    if n > 0.0 {
        for c in &mut counts {
            *c /= n;
        }
    }
    counts
}

pub fn build_features(frame: &FeatureFrame<'_>) -> Vec<f64> {
    let mut f = vec![0.0; N_FEATURES];
    let (body, yaw_err, dist) = crate::planner::body_error(frame.state, frame.goal);
    f[0] = body.x / 5.0;
    f[1] = body.y / 5.0;
    f[2] = body.z / 5.0;
    f[3] = dist / 10.0;
    f[4] = yaw_err / std::f64::consts::PI;
    f[5] = frame.state.yaw.sin();
    f[6] = frame.state.yaw.cos();
    f[7] = frame.state.position.z / 10.0;
    f[8] = frame.state.velocity_ned.norm() / 3.0;
    if frame.intent_type < 9 {
        f[9 + frame.intent_type] = 1.0;
    }
    f[18] = (frame.magnitude / 5.0).clamp(-2.0, 2.0);
    f[19] = frame.side_sign;
    f[20] = frame.progress.clamp(0.0, 1.0);
    f[21] = frame.completed_fraction.clamp(0.0, 1.0);
    f[22] = frame.time_fraction.clamp(0.0, 1.0);
    if let Some((bearing, range)) = frame.landmark {
        f[23] = bearing.sin();
        f[24] = bearing.cos();
        f[25] = (range / 20.0).clamp(0.0, 2.0);
        f[26] = 1.0;
    }
    let hash = text_hash(frame.instruction);
    f[27..43].copy_from_slice(&hash);
    f[43] = frame.prev_action.dx / 0.35;
    f[44] = frame.prev_action.dy / 0.22;
    f[45] = frame.prev_action.dz / 0.18;
    f[46] = frame.prev_action.dyaw / 0.12;
    f[47] = 1.0;
    f
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearPolicy {
    pub w: Vec<Vec<f64>>, // 4 x N_FEATURES
    pub b: [f64; 4],
    pub scales: [f64; 4],
    pub sigma: [f64; 4],
}

#[derive(Debug, Clone, Copy)]
pub struct ActionSample {
    pub mean: [f64; 4],
    pub action: Delta4,
    pub logprob: f64,
}

impl LinearPolicy {
    pub fn zeros() -> Self {
        Self {
            w: vec![vec![0.0; N_FEATURES]; 4],
            b: [0.0; 4],
            scales: [0.35, 0.22, 0.18, 0.12],
            sigma: [0.045, 0.030, 0.030, 0.022],
        }
    }

    /// M0: calibrated interface, no learned UAV policy. The mean command is a
    /// small forward bias so the base model is not a pure hover controller.
    pub fn m0() -> Self {
        let mut p = Self::zeros();
        p.b[0] = 0.01;
        p
    }

    pub fn predict(&self, features: &[f64]) -> [f64; 4] {
        let mut out = [0.0; 4];
        for d in 0..4 {
            let z = self.b[d]
                + self.w[d]
                    .iter()
                    .zip(features.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>();
            out[d] = self.scales[d] * z.tanh();
        }
        out
    }

    pub fn sample<R: Rng>(&self, features: &[f64], rng: &mut R) -> ActionSample {
        let mean = self.predict(features);
        let mut action = [0.0; 4];
        let mut logprob = 0.0;
        let two_pi = (2.0 * std::f64::consts::PI).sqrt().ln();
        for d in 0..4 {
            let n: f64 = {
                let u1 = rng.gen::<f64>().max(1e-12);
                let u2 = rng.gen::<f64>();
                (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
            };
            let a = mean[d] + self.sigma[d] * n;
            action[d] = a.clamp(-self.scales[d] * 1.2, self.scales[d] * 1.2);
            let z = (a - mean[d]) / self.sigma[d];
            logprob += -0.5 * z * z - self.sigma[d].ln() - two_pi;
        }
        ActionSample {
            mean,
            action: Delta4::new(action[0], action[1], action[2], action[3]),
            logprob,
        }
    }

    pub fn logprob(&self, features: &[f64], action: Delta4) -> (f64, [f64; 4]) {
        let mean = self.predict(features);
        let a = [action.dx, action.dy, action.dz, action.dyaw];
        let mut lp = 0.0;
        let two_pi = (2.0 * std::f64::consts::PI).sqrt().ln();
        for d in 0..4 {
            let z = (a[d] - mean[d]) / self.sigma[d];
            lp += -0.5 * z * z - self.sigma[d].ln() - two_pi;
        }
        (lp, mean)
    }

    /// d(loss)/d(pre-activation) for a per-dimension gradient on the mean.
    pub fn grad_from_mean(&self, features: &[f64], d_mean: &[f64; 4]) -> (Vec<Vec<f64>>, [f64; 4]) {
        let mean = self.predict(features);
        let mut gw = vec![vec![0.0; N_FEATURES]; 4];
        let mut gb = [0.0; 4];
        for d in 0..4 {
            let z = (mean[d] / self.scales[d]).clamp(-0.999999, 0.999999);
            let dz = d_mean[d] * self.scales[d] * (1.0 - z * z);
            gb[d] = dz;
            for (i, x) in features.iter().enumerate() {
                gw[d][i] = dz * x;
            }
        }
        (gw, gb)
    }

    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }
}

pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub t: u64,
    m: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
    mb: [f64; 4],
    vb: [f64; 4],
}

impl Adam {
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m: vec![vec![0.0; N_FEATURES]; 4],
            v: vec![vec![0.0; N_FEATURES]; 4],
            mb: [0.0; 4],
            vb: [0.0; 4],
        }
    }
    pub fn step(&mut self, policy: &mut LinearPolicy, gw: &[Vec<f64>], gb: &[f64; 4], l2: f64) {
        self.t += 1;
        let c1 = 1.0 - self.beta1.powi(self.t as i32);
        let c2 = 1.0 - self.beta2.powi(self.t as i32);
        for d in 0..4 {
            for i in 0..N_FEATURES {
                let g = gw[d][i] + l2 * policy.w[d][i];
                self.m[d][i] = self.beta1 * self.m[d][i] + (1.0 - self.beta1) * g;
                self.v[d][i] = self.beta2 * self.v[d][i] + (1.0 - self.beta2) * g * g;
                let mhat = self.m[d][i] / c1;
                let vhat = self.v[d][i] / c2;
                policy.w[d][i] -= self.lr * mhat / (vhat.sqrt() + self.eps);
            }
            let g = gb[d];
            self.mb[d] = self.beta1 * self.mb[d] + (1.0 - self.beta1) * g;
            self.vb[d] = self.beta2 * self.vb[d] + (1.0 - self.beta2) * g * g;
            let mhat = self.mb[d] / c1;
            let vhat = self.vb[d] / c2;
            policy.b[d] -= self.lr * mhat / (vhat.sqrt() + self.eps);
        }
    }
}

pub fn body_goal_error(state: SimState, goal: Goal) -> (Vec3, f64) {
    let body = ned_to_body(goal.position.sub(state.position), state.yaw);
    (body, wrap_angle(goal.yaw - state.yaw))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_shape_and_hash() {
        let frame = FeatureFrame {
            instruction: "Move forward 3.0 metres",
            intent_type: 0,
            magnitude: 3.0,
            side_sign: 0.0,
            progress: 0.2,
            completed_fraction: 0.0,
            time_fraction: 0.1,
            state: SimState {
                sim_time_s: 1.0,
                position: Vec3::new(0.0, 0.0, -10.0),
                velocity_ned: Vec3::zero(),
                yaw: 0.0,
            },
            goal: Goal { position: Vec3::new(3.0, 0.0, -10.0), yaw: 0.0 },
            landmark: Some((0.2, 6.0)),
            prev_action: Delta4::zero(),
        };
        let f = build_features(&frame);
        assert_eq!(f.len(), N_FEATURES);
        assert!((f[0] - 0.6).abs() < 1e-9);
        assert_eq!(f[9], 1.0);
        assert_eq!(f[26], 1.0);
        let h = text_hash("Move forward 3.0 metres");
        assert!((h.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }
}
