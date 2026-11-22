//! Reference-trajectory reward used by EG-GRPO rollout collection.
//!
//! Port of `integrations/uavflow_sim/rl/reward.py`. Coordinates are centimetres
//! in task JSON and episode logs; all returned distances are metres.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RolloutReward {
    pub reward: f64,
    pub progress: f64,
    pub cross_track_m: f64,
    pub terminal_m: f64,
}

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

pub fn reference_path_cm(task: &serde_json::Value) -> Result<Vec<[f64; 3]>> {
    let arr = task
        .get("reference_path_raw")
        .and_then(|v| v.as_array())
        .context("task has no reference_path_raw")?;
    let mut out = Vec::new();
    for pose in arr {
        let p = pose.as_array().context("reference pose is not an array")?;
        if p.len() < 3 {
            anyhow::bail!("reference pose has fewer than 3 coordinates");
        }
        out.push([
            p[0].as_f64().unwrap_or(0.0),
            p[1].as_f64().unwrap_or(0.0),
            p[2].as_f64().unwrap_or(0.0),
        ]);
    }
    if out.is_empty() {
        anyhow::bail!("empty reference path");
    }
    Ok(out)
}

pub fn rollout_reward(
    world_xyz_cm: &[[f64; 3]],
    task: &serde_json::Value,
) -> Result<RolloutReward> {
    let refp = reference_path_cm(task)?;
    if world_xyz_cm.is_empty() {
        return Ok(RolloutReward {
            reward: -10.0,
            progress: 0.0,
            cross_track_m: 10.0,
            terminal_m: 10.0,
        });
    }
    let mut nn_max = 0usize;
    let mut cross_sum = 0.0;
    for p in world_xyz_cm {
        let mut best = f64::INFINITY;
        let mut best_i = 0usize;
        for (i, r) in refp.iter().enumerate() {
            let d = dist(p, r);
            if d < best {
                best = d;
                best_i = i;
            }
        }
        cross_sum += best;
        nn_max = nn_max.max(best_i);
    }
    let progress = nn_max as f64 / (refp.len() - 1).max(1) as f64;
    let cross_track_m = cross_sum / world_xyz_cm.len() as f64 / 100.0;
    let terminal_m = dist(world_xyz_cm.last().unwrap(), refp.last().unwrap()) / 100.0;
    Ok(RolloutReward {
        reward: 2.0 * progress - 0.5 * cross_track_m - terminal_m,
        progress,
        cross_track_m,
        terminal_m,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn straight_path_reward() {
        let task = serde_json::json!({"reference_path_raw": [[0,0,0],[100,0,0],[200,0,0]]});
        let r = rollout_reward(
            &[[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 0.0, 0.0]],
            &task,
        )
        .unwrap();
        assert!((r.progress - 1.0).abs() < 1e-9);
        assert!((r.cross_track_m - 0.0).abs() < 1e-9);
        assert!((r.terminal_m - 0.0).abs() < 1e-9);
        assert!((r.reward - 2.0).abs() < 1e-9);
    }
    #[test]
    fn empty_rollout_penalty() {
        let task = serde_json::json!({"reference_path_raw": [[0,0,0],[1,0,0]]});
        let r = rollout_reward(&[], &task).unwrap();
        assert_eq!(r.reward, -10.0);
    }
}
