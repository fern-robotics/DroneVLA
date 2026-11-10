//! Actual UAV-Flow-Sim trajectories converted by scripts/prepare_uavflow.py.
//! The official OpenVLA-UAV loader defines the action convention; the Python
//! converter reproduces it and stores metre/radian delta4 values.

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UavEpisode {
    pub id: String,
    pub instruction: String,
    pub intent_tag: String,
    pub params: serde_json::Value,
    pub positions: Vec<[f64; 3]>,
    pub yaw: Vec<f64>,
    pub actions: Vec<[f64; 4]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UavStats {
    pub episodes_selected: usize,
    pub frames: usize,
    pub q01: Vec<f64>,
    pub q99: Vec<f64>,
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    pub abs_mean: Vec<f64>,
}

pub fn load(path: &Path) -> anyhow::Result<Vec<UavEpisode>> {
    let text = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(line)?);
    }
    Ok(out)
}

pub fn load_stats(path: &Path) -> anyhow::Result<UavStats> {
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

impl UavEpisode {
    /// Instruction tag to the nine DroneVL intent indices used by the policy.
    pub fn intent_type(&self) -> usize {
        match self.intent_tag.as_str() {
            "move" => 0,
            "yaw" => 1,
            "altitude" => 2,
            "hold" => 3,
            "approach" => 4,
            "pass" => 5,
            "orbit" => 6,
            "hold_at" => 7,
            "return" => 8,
            _ => 0,
        }
    }
    pub fn magnitude(&self) -> f64 {
        self.params
            .get("distance_m")
            .and_then(|v| v.as_f64())
            .or_else(|| self.params.get("angle_deg").and_then(|v| v.as_f64()))
            .unwrap_or(3.0)
    }
    pub fn side_sign(&self) -> f64 {
        self.params.get("side").and_then(|v| v.as_f64()).unwrap_or(0.0)
    }
}
