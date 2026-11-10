//! SFT on actual UAV-Flow-Sim trajectories and GRPO on AirSimNH rollouts.

use crate::action::{ActionCodec, Delta4};
use crate::geom::{body_to_ned, Vec3};
use crate::policy::{build_features, Adam, FeatureFrame, LinearPolicy, N_FEATURES};
use crate::runner::{EpisodeRun, RunConfig};
use crate::sim::SimState;
use crate::task::TaskEpisode;
use crate::uavflow::UavEpisode;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct SftSample {
    pub features: Vec<f64>,
    pub target: [f64; 4],
    pub episode_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftLog {
    pub epochs: usize,
    pub train_samples: usize,
    pub val_samples: usize,
    pub train_loss: Vec<f64>,
    pub val_loss: Vec<f64>,
}

pub fn build_sft_samples(episodes: &[UavEpisode], stride: usize) -> Vec<SftSample> {
    let mut out = Vec::new();
    for ep in episodes {
        let n = ep.actions.len();
        if n < 6 {
            continue;
        }
        for t in (0..n - 1).step_by(stride.max(1)) {
            let goal_i = (t + 5).min(n - 1);
            let prev = if t > 0 {
                Delta4::new(ep.actions[t - 1][0], ep.actions[t - 1][1], ep.actions[t - 1][2], ep.actions[t - 1][3])
            } else {
                Delta4::zero()
            };
            let velocity = body_to_ned(
                Vec3::new(ep.actions[t][0], ep.actions[t][1], ep.actions[t][2]).scale(10.0),
                ep.yaw[t],
            );
            let state = SimState {
                sim_time_s: t as f64 * 0.1,
                position: Vec3::from_array(ep.positions[t]),
                velocity_ned: velocity,
                yaw: ep.yaw[t],
            };
            let goal = crate::planner::Goal {
                position: Vec3::from_array(ep.positions[goal_i]),
                yaw: ep.yaw[goal_i],
            };
            let progress = t as f64 / (n - 1) as f64;
            let f = build_features(&FeatureFrame {
                instruction: &ep.instruction,
                intent_type: ep.intent_type(),
                magnitude: ep.magnitude(),
                side_sign: ep.side_sign(),
                progress,
                completed_fraction: 0.0,
                time_fraction: progress,
                state,
                goal,
                landmark: None,
                prev_action: prev,
            });
            out.push(SftSample {
                features: f,
                target: ep.actions[t],
                episode_id: ep.id.clone(),
            });
        }
    }
    out
}

pub fn train_sft(
    init: &LinearPolicy,
    samples: &[SftSample],
    epochs: usize,
    batch: usize,
    lr: f64,
    l2: f64,
    seed: u64,
) -> (LinearPolicy, SftLog) {
    let mut policy = init.clone();
    let mut adam = Adam::new(lr);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut idx: Vec<usize> = (0..samples.len()).collect();
    // Deterministic episode-level split for validation.
    let val_ids: std::collections::HashSet<String> = samples
        .iter()
        .filter(|s| fnv(&s.episode_id) % 10 == 0)
        .map(|s| s.episode_id.clone())
        .collect();
    let train: Vec<usize> = idx.iter().cloned().filter(|i| !val_ids.contains(&samples[*i].episode_id)).collect();
    let val: Vec<usize> = idx.iter().cloned().filter(|i| val_ids.contains(&samples[*i].episode_id)).collect();
    let mut train_loss = Vec::new();
    let mut val_loss = Vec::new();
    for _ in 0..epochs {
        idx = train.clone();
        idx.shuffle(&mut rng);
        let mut epoch_loss = 0.0;
        let mut nb = 0;
        for chunk in idx.chunks(batch.max(1)) {
            let mut gw = vec![vec![0.0; N_FEATURES]; 4];
            let mut gb = [0.0; 4];
            for &i in chunk {
                let s = &samples[i];
                let pred = policy.predict(&s.features);
                let mut dmean = [0.0; 4];
                for d in 0..4 {
                    let e = (pred[d] - s.target[d]) / policy.scales[d];
                    epoch_loss += e * e;
                    dmean[d] = 2.0 * e / (policy.scales[d] * chunk.len() as f64);
                }
                let (g, b) = policy.grad_from_mean(&s.features, &dmean);
                for d in 0..4 {
                    gb[d] += b[d];
                    for j in 0..N_FEATURES {
                        gw[d][j] += g[d][j];
                    }
                }
            }
            adam.step(&mut policy, &gw, &gb, l2);
            nb += 1;
        }
        train_loss.push(epoch_loss / (train.len().max(1) as f64));
        let mut vl = 0.0;
        for &i in &val {
            let s = &samples[i];
            let pred = policy.predict(&s.features);
            for d in 0..4 {
                let e = (pred[d] - s.target[d]) / policy.scales[d];
                vl += e * e;
            }
        }
        val_loss.push(vl / (val.len().max(1) as f64));
        let _ = nb;
    }
    (
        policy,
        SftLog {
            epochs,
            train_samples: train.len(),
            val_samples: val.len(),
            train_loss,
            val_loss,
        },
    )
}

fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupLog {
    pub group: usize,
    pub reward_mean: f64,
    pub reward_std: f64,
    pub success_rate: f64,
    pub intent_score: f64,
    pub skipped: bool,
    pub ratio_mean: f64,
    pub loss: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpoLog {
    pub groups: Vec<GroupLog>,
    pub effective_updates: usize,
    pub skipped_groups: usize,
}

pub fn grpo_reward(s: &crate::runner::EpisodeSummary) -> f64 {
    let success = if s.success { 1.0 } else { 0.0 };
    let collision = if s.had_collision { 1.0 } else { 0.0 };
    1.0 * success + 0.5 * s.ordered_completion + 1.0 * s.intent_score
        - 1.0 * collision
        - 0.25 * s.invalid_action_rate
        - 0.1 * s.safety_intervention_rate
}

pub fn train_grpo(
    init: &LinearPolicy,
    reference: &LinearPolicy,
    episodes: &[TaskEpisode],
    codec: &ActionCodec,
    groups: usize,
    rollouts: usize,
    lr: f64,
    clip_eps: f64,
    beta: f64,
    seed: u64,
) -> (LinearPolicy, GrpoLog) {
    let mut policy = init.clone();
    let mut adam = Adam::new(lr);
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x9e3779b97f4a7c15);
    let mut log = GrpoLog {
        groups: Vec::new(),
        effective_updates: 0,
        skipped_groups: 0,
    };
    for g in 0..groups {
        let ep = &episodes[rng.gen_range(0..episodes.len())];
        let mut results = Vec::new();
        for r in 0..rollouts {
            let mut cfg = RunConfig::rollout("surrogate", "M2", seed, g as u64 + 1);
            cfg.policy_salt = (g * 97 + r) as u64 + 1;
            let result = EpisodeRun {
                episode: ep,
                policy: &policy,
                codec,
                config: &cfg,
            }
            .execute();
            results.push(result);
        }
        let rewards: Vec<f64> = results.iter().map(|r| grpo_reward(&r.summary)).collect();
        let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
        let var = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rewards.len() as f64;
        let std = var.sqrt();
        let success_rate = results.iter().filter(|r| r.summary.success).count() as f64 / rollouts as f64;
        let intent = results.iter().map(|r| r.summary.intent_score).sum::<f64>() / rollouts as f64;
        if std < 1e-8 {
            log.skipped_groups += 1;
            log.groups.push(GroupLog {
                group: g,
                reward_mean: mean,
                reward_std: std,
                success_rate,
                intent_score: intent,
                skipped: true,
                ratio_mean: 1.0,
                loss: 0.0,
            });
            continue;
        }
        let advantages: Vec<f64> = rewards.iter().map(|r| (r - mean) / std).collect();
        let mut gw = vec![vec![0.0; N_FEATURES]; 4];
        let mut gb = [0.0; 4];
        let mut ratio_sum = 0.0;
        let mut ratio_n = 0usize;
        let mut loss = 0.0;
        for (ri, result) in results.iter().enumerate() {
            let adv = advantages[ri];
            let t_i = result.traces.len().max(1) as f64;
            for tr in &result.traces {
                let action = Delta4::new(tr.action[0], tr.action[1], tr.action[2], tr.action[3]);
                let (new_lp, mean_now) = policy.logprob(&tr.features, action);
                let ratio = (new_lp - tr.logprob).exp().clamp(0.0, 10.0);
                ratio_sum += ratio;
                ratio_n += 1;
                let clipped = if adv >= 0.0 {
                    ratio.min(1.0 + clip_eps)
                } else {
                    ratio.max(1.0 - clip_eps)
                };
                let active = if adv >= 0.0 { ratio < 1.0 + clip_eps } else { ratio > 1.0 - clip_eps };
                let coef = if active { -adv * ratio } else { 0.0 };
                loss += -(ratio * adv).min(clipped * adv);
                let mean_ref = reference.predict(&tr.features);
                let mut dmean = [0.0; 4];
                for d in 0..4 {
                    let sigma2 = policy.sigma[d] * policy.sigma[d];
                    let pg = coef * (tr.action[d] - mean_now[d]) / sigma2;
                    let kl = beta * (mean_now[d] - mean_ref[d]) / sigma2;
                    dmean[d] = (pg + kl) / (rollouts as f64 * t_i);
                }
                let (gr, br) = policy.grad_from_mean(&tr.features, &dmean);
                for d in 0..4 {
                    gb[d] += br[d];
                    for j in 0..N_FEATURES {
                        gw[d][j] += gr[d][j];
                    }
                }
            }
        }
        adam.step(&mut policy, &gw, &gb, 0.0);
        log.effective_updates += 1;
        log.groups.push(GroupLog {
            group: g,
            reward_mean: mean,
            reward_std: std,
            success_rate,
            intent_score: intent,
            skipped: false,
            ratio_mean: ratio_sum / ratio_n.max(1) as f64,
            loss,
        });
    }
    (policy, log)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sft_reduces_loss_on_uavflow() {
        let path = std::path::Path::new("data/uavflow_sim.jsonl");
        if !path.exists() {
            return;
        }
        let eps = crate::uavflow::load(path).unwrap();
        let samples = build_sft_samples(&eps, 10);
        assert!(!samples.is_empty());
        let (p, log) = train_sft(&LinearPolicy::m0(), &samples, 5, 32, 0.01, 1e-4, 0);
        assert!(*log.val_loss.last().unwrap() <= *log.val_loss.first().unwrap() * 1.05);
        assert!(p.predict(&samples[0].features).iter().all(|v| v.is_finite()));
    }
}
