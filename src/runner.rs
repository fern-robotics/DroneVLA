//! Closed-loop episode runner: render -> perceive -> plan -> policy -> gate ->
//! execute -> evaluate. Simulation is paused during inference.

use crate::action::{ActionCodec, Delta4};
use crate::geom::{ned_to_body, Camera, Vec3};
use crate::planner::{self, Goal};
use crate::policy::{build_features, FeatureFrame, LinearPolicy};
use crate::safety::{GateDecision, GateDisposition, GateObservation, SafetyGate};
use crate::scene::{perceive, PerceptionConfig};
use crate::sim::Simulator;
use crate::task::TaskEpisode;
use crate::tracker::MissionTracker;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

pub const STEP_DT: f64 = 0.1;
pub const CAMERA_W: usize = 96;
pub const CAMERA_H: usize = 72;
pub const CAMERA_FOV: f64 = 100.0 * std::f64::consts::PI / 180.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    pub arm: String,
    pub condition: String,
    pub train_seed: u64,
    pub sample_policy: bool,
    pub noise_salt: u64,
    pub policy_salt: u64,
    pub store_frames: bool,
    pub frame_every: u32,
    pub out_dir: Option<PathBuf>,
}

impl RunConfig {
    pub fn eval(arm: &str, condition: &str, seed: u64) -> Self {
        Self {
            arm: arm.into(),
            condition: condition.into(),
            train_seed: seed,
            sample_policy: false,
            noise_salt: 0,
            policy_salt: 0,
            store_frames: false,
            frame_every: 10,
            out_dir: None,
        }
    }
    pub fn rollout(arm: &str, condition: &str, seed: u64, salt: u64) -> Self {
        Self {
            arm: arm.into(),
            condition: condition.into(),
            train_seed: seed,
            sample_policy: true,
            noise_salt: salt,
            policy_salt: salt,
            store_frames: false,
            frame_every: 10,
            out_dir: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStep {
    pub decision_id: u32,
    pub features: Vec<f64>,
    pub mean: [f64; 4],
    pub action: [f64; 4],
    pub logprob: f64,
    pub intent_type: usize,
    pub progress: f64,
    pub sim_time_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRecord {
    pub decision_id: u32,
    pub sim_time_s: f64,
    pub position: [f64; 3],
    pub yaw_rad: f64,
    pub intent_type: String,
    pub intent_progress: f64,
    pub goal: [f64; 3],
    pub goal_yaw: f64,
    pub landmark_present: bool,
    pub landmark_range_m: Option<f64>,
    pub mean_action: Delta4,
    pub executed: Delta4,
    pub gate: GateDisposition,
    pub gate_reasons: Vec<String>,
    pub tokens: Vec<u32>,
    pub codec: String,
    pub quantization_error: f64,
    pub inference_ms: f64,
    pub decision_ms: f64,
    pub visible_markers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentOutcome {
    pub intent_type: String,
    pub progress: f64,
    pub satisfied: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeSummary {
    pub episode_id: String,
    pub family: String,
    pub split: String,
    pub ood: bool,
    pub arm: String,
    pub condition: String,
    pub train_seed: u64,
    pub success: bool,
    pub terminal_reason: String,
    pub failure_mode: Option<String>,
    pub intent_score: f64,
    pub intents_completed: usize,
    pub intents_total: usize,
    pub ordered_completion: f64,
    pub intents: Vec<IntentOutcome>,
    pub decisions: u32,
    pub collisions: u32,
    pub had_collision: bool,
    pub invalid_actions: u32,
    pub invalid_action_rate: f64,
    pub safety_interventions: u32,
    pub safety_rejections: u32,
    pub safety_clips: u32,
    pub safety_intervention_rate: f64,
    pub path_length_m: f64,
    pub reference_path_m: f64,
    pub successful_path_efficiency: Option<f64>,
    pub mission_sim_time_s: f64,
    pub mission_wall_time_s: f64,
    pub final_position: [f64; 3],
    pub inference_ms_median: f64,
    pub inference_ms_p95: f64,
    pub decision_ms_median: f64,
    pub total_tokens: u32,
    pub mean_quantization_error: f64,
}

pub struct RunResult {
    pub summary: EpisodeSummary,
    pub decisions: Vec<DecisionRecord>,
    pub traces: Vec<TraceStep>,
}

pub struct EpisodeRun<'a> {
    pub episode: &'a TaskEpisode,
    pub policy: &'a LinearPolicy,
    pub codec: &'a ActionCodec,
    pub config: &'a RunConfig,
}

impl<'a> EpisodeRun<'a> {
    pub fn execute(&self) -> RunResult {
        let ep = self.episode;
        let scene = ep.scene();
        let camera = Camera::new(CAMERA_W, CAMERA_H, CAMERA_FOV);
        let mut sim = Simulator::new(ep);
        let mut tracker = MissionTracker::new(ep);
        let gate = SafetyGate::new(ep.limits.clone());
        let perception_cfg = PerceptionConfig::default();
        let start = Instant::now();
        let mut decisions = Vec::new();
        let mut traces = Vec::new();
        let mut prev_action = Delta4::zero();
        let mut landmark_memory: Option<Vec3> = None;
        let mut last_intent_index = usize::MAX;
        let mut collision_ids: HashSet<String> = HashSet::new();
        let mut collisions = 0u32;
        let mut invalid = 0u32;
        let mut terminal_reason = "DECISION_LIMIT".to_string();
        let mut last_pos = sim.state.position;
        let mut path_length = 0.0;

        loop {
            if decisions.len() as u32 >= ep.limits.max_decisions {
                terminal_reason = "DECISION_LIMIT".into();
                break;
            }
            if sim.state.sim_time_s > ep.limits.max_sim_time_s {
                terminal_reason = "TIME_LIMIT".into();
                break;
            }
            if tracker.is_complete() {
                terminal_reason = "SUCCESS".into();
                break;
            }
            if tracker.index != last_intent_index {
                landmark_memory = None;
                last_intent_index = tracker.index;
            }
            let decision_id = decisions.len() as u32;
            let decision_start = Instant::now();
            let frame = scene.render(&camera, sim.state.position, sim.state.yaw);
            let front_clearance = front_clearance(&frame);
            let noise_seed = ep
                .seed
                .wrapping_add((decision_id as u64).wrapping_mul(0x9e3779b97f4a7c15))
                .wrapping_add(self.config.noise_salt.wrapping_mul(0xbf58476d1ce4e5b9));
            let mut perception_rng = ChaCha8Rng::seed_from_u64(noise_seed);
            let detections = perceive(
                &frame,
                &camera,
                sim.state.position,
                sim.state.yaw,
                &perception_cfg,
                &mut perception_rng,
            );
            let current_intent = tracker.current().cloned();
            let intent_type = current_intent.as_ref().map(|i| i.type_index()).unwrap_or(8);
            let progress = tracker.current_progress();
            let mut landmark_rel: Option<(f64, f64)> = None;
            if let Some(desc) = current_intent.as_ref().and_then(|i| i.landmark()) {
                if let Some(det) = detections
                    .iter()
                    .filter(|d| d.color == desc.color && d.shape == desc.shape)
                    .min_by(|a, b| a.range_m.partial_cmp(&b.range_m).unwrap())
                {
                    landmark_memory = Some(det.world_position);
                    landmark_rel = Some((det.bearing_rad, det.range_m));
                }
                if landmark_rel.is_none() {
                    if let Some(mem) = landmark_memory {
                        let rel = ned_to_body(mem.sub(sim.state.position), sim.state.yaw);
                        let bearing = rel.y.atan2(rel.x);
                        let range = rel.horizontal_norm();
                        landmark_rel = Some((bearing, range));
                    }
                }
            }
            let rt = tracker.runtime().cloned().unwrap();
            let goal = current_intent
                .as_ref()
                .map(|i| planner::plan(i, &rt, ep, sim.state, landmark_memory))
                .unwrap_or(Goal { position: sim.state.position, yaw: sim.state.yaw });
            let features = build_features(&FeatureFrame {
                instruction: &ep.instruction,
                intent_type,
                magnitude: current_intent.as_ref().map(|i| i.magnitude()).unwrap_or(0.0),
                side_sign: current_intent.as_ref().map(|i| i.side_sign()).unwrap_or(0.0),
                progress,
                completed_fraction: tracker.ordered_completion(),
                time_fraction: sim.state.sim_time_s / ep.limits.max_sim_time_s.max(1e-6),
                state: sim.state,
                goal,
                landmark: landmark_rel,
                prev_action,
            });
            let inference_start = Instant::now();
            let mut rng = ChaCha8Rng::seed_from_u64(
                noise_seed ^ 0x517cc1b727220a95 ^ self.config.policy_salt.wrapping_mul(0x9e3779b97f4a7c15),
            );
            let sample = if self.config.sample_policy {
                self.policy.sample(&features, &mut rng)
            } else {
                let mean = self.policy.predict(&features);
                let (lp, _) = self.policy.logprob(&features, Delta4::new(mean[0], mean[1], mean[2], mean[3]));
                crate::policy::ActionSample {
                    mean,
                    action: Delta4::new(mean[0], mean[1], mean[2], mean[3]),
                    logprob: lp,
                }
            };
            let inference_ms = inference_start.elapsed().as_secs_f64() * 1000.0;
            let horizon = self.codec.exec_horizon();
            let chunk = vec![sample.action; horizon.max(1)];
            let plan = self.codec.quantize(&chunk);
            let mut first_gate = GateDecision {
                disposition: GateDisposition::Accepted,
                reasons: Vec::new(),
                action: Some(sample.action),
            };
            let mut executed = Delta4::zero();
            let mut executed_any = false;
            for (i, a) in plan.executed().iter().enumerate() {
                let gate_obs = GateObservation {
                    position: sim.state.position,
                    yaw: sim.state.yaw,
                    sim_time_s: sim.state.sim_time_s,
                    sensor_timestamp_s: Some(sim.state.sim_time_s),
                    front_clearance_m: front_clearance,
                };
                let age = decision_start.elapsed().as_secs_f64();
                let d = gate.check(*a, &gate_obs, age);
                if i == 0 {
                    first_gate = d.clone();
                    executed = d.action.unwrap_or(Delta4::zero());
                }
                match d.action {
                    Some(cmd) => {
                        executed_any = true;
                        let out = sim.step(cmd, STEP_DT);
                        path_length += last_pos.distance(sim.state.position);
                        last_pos = sim.state.position;
                        if let Some(id) = out.collision_id {
                            if collision_ids.insert(id) {
                                collisions += 1;
                            }
                        }
                        if !ep.limits.contains(sim.state.position) {
                            tracker.update(sim.state, STEP_DT);
                            terminal_reason = "GEOFENCE_VIOLATION".into();
                            break;
                        }
                        if tracker.update(sim.state, STEP_DT) {
                            terminal_reason = "SUCCESS".into();
                            break;
                        }
                    }
                    None => {
                        invalid += 1;
                        let out = sim.step(Delta4::zero(), STEP_DT);
                        path_length += last_pos.distance(sim.state.position);
                        last_pos = sim.state.position;
                        if let Some(id) = out.collision_id {
                            if collision_ids.insert(id) {
                                collisions += 1;
                            }
                        }
                        if tracker.update(sim.state, STEP_DT) {
                            terminal_reason = "SUCCESS".into();
                            break;
                        }
                    }
                }
            }
            let _ = executed_any;
            let decision_ms = decision_start.elapsed().as_secs_f64() * 1000.0;
            decisions.push(DecisionRecord {
                decision_id,
                sim_time_s: sim.state.sim_time_s,
                position: sim.state.position.to_array(),
                yaw_rad: sim.state.yaw,
                intent_type: current_intent.as_ref().map(|i| i.type_name().to_string()).unwrap_or_default(),
                intent_progress: progress,
                goal: goal.position.to_array(),
                goal_yaw: goal.yaw,
                landmark_present: landmark_rel.is_some(),
                landmark_range_m: landmark_rel.map(|(_, r)| r),
                mean_action: Delta4::new(sample.mean[0], sample.mean[1], sample.mean[2], sample.mean[3]),
                executed,
                gate: first_gate.disposition,
                gate_reasons: first_gate.reasons.clone(),
                tokens: plan.tokens.clone(),
                codec: plan.codec.clone(),
                quantization_error: plan.max_quantization_error,
                inference_ms,
                decision_ms,
                visible_markers: detections.len(),
            });
            traces.push(TraceStep {
                decision_id,
                features,
                mean: sample.mean,
                action: [sample.action.dx, sample.action.dy, sample.action.dz, sample.action.dyaw],
                logprob: sample.logprob,
                intent_type,
                progress,
                sim_time_s: sim.state.sim_time_s,
            });
            prev_action = executed;
            if terminal_reason == "SUCCESS" || terminal_reason == "GEOFENCE_VIOLATION" {
                break;
            }
            if collisions > 0 && ep.limits.terminate_on_collision {
                terminal_reason = "COLLISION".into();
                break;
            }
        }
        if tracker.is_complete() {
            terminal_reason = "SUCCESS".into();
        }
        let wall = start.elapsed().as_secs_f64();
        let success = tracker.is_complete();
        let per_intent: Vec<IntentOutcome> = ep
            .intents
            .iter()
            .enumerate()
            .map(|(i, it)| {
                let progress = tracker.runtimes.get(i).map(|r| r.progress).unwrap_or(0.0);
                IntentOutcome {
                    intent_type: it.type_name().to_string(),
                    progress: if success { 1.0 } else { progress },
                    satisfied: success || i < tracker.completed,
                }
            })
            .collect();
        let interventions = decisions.iter().filter(|d| d.gate != GateDisposition::Accepted).count();
        let rejects = decisions.iter().filter(|d| d.gate == GateDisposition::Rejected).count();
        let clips = decisions.iter().filter(|d| d.gate == GateDisposition::Clipped).count();
        let inference: Vec<f64> = decisions.iter().map(|d| d.inference_ms).collect();
        let decision: Vec<f64> = decisions.iter().map(|d| d.decision_ms).collect();
        let reference = ep.reference_path_m;
        let efficiency = if success && reference > 0.0 && path_length > 0.0 {
            Some(reference / path_length)
        } else {
            None
        };
        let failure_mode = if success {
            None
        } else {
            Some(match terminal_reason.as_str() {
                "COLLISION" => "COLLISION".to_string(),
                "GEOFENCE_VIOLATION" => "GEOFENCE".to_string(),
                "TIME_LIMIT" | "DECISION_LIMIT" => format!(
                    "{}_INCOMPLETE",
                    tracker.current().map(|i| i.type_name()).unwrap_or("mission")
                ),
                _ => "UNKNOWN".to_string(),
            })
        };
        let summary = EpisodeSummary {
            episode_id: ep.id.clone(),
            family: ep.family.clone(),
            split: ep.split.clone(),
            ood: ep.ood,
            arm: self.config.arm.clone(),
            condition: self.config.condition.clone(),
            train_seed: self.config.train_seed,
            success,
            terminal_reason,
            failure_mode,
            intent_score: tracker.intent_score(),
            intents_completed: tracker.completed,
            intents_total: ep.intents.len(),
            ordered_completion: tracker.ordered_completion(),
            intents: per_intent,
            decisions: decisions.len() as u32,
            collisions,
            had_collision: collisions > 0,
            invalid_actions: invalid,
            invalid_action_rate: if decisions.is_empty() { 0.0 } else { invalid as f64 / decisions.len() as f64 },
            safety_interventions: interventions as u32,
            safety_rejections: rejects as u32,
            safety_clips: clips as u32,
            safety_intervention_rate: if decisions.is_empty() { 0.0 } else { interventions as f64 / decisions.len() as f64 },
            path_length_m: path_length,
            reference_path_m: reference,
            successful_path_efficiency: efficiency,
            mission_sim_time_s: sim.state.sim_time_s,
            mission_wall_time_s: wall,
            final_position: sim.state.position.to_array(),
            inference_ms_median: median(&inference),
            inference_ms_p95: p95(&inference),
            decision_ms_median: median(&decision),
            total_tokens: decisions.iter().map(|d| d.tokens.len() as u32).sum(),
            mean_quantization_error: if decisions.is_empty() { 0.0 } else { decisions.iter().map(|d| d.quantization_error).sum::<f64>() / decisions.len() as f64 },
        };
        RunResult { summary, decisions, traces }
    }
}

pub fn front_clearance(frame: &crate::scene::Frame) -> Option<f64> {
    let x0 = frame.width * 2 / 5;
    let x1 = frame.width * 3 / 5;
    let y0 = frame.height / 3;
    let y1 = frame.height * 2 / 3;
    let mut best = f64::INFINITY;
    for y in y0..y1 {
        for x in x0..x1 {
            let d = frame.depth[y * frame.width + x] as f64;
            if d.is_finite() && d < best {
                best = d;
            }
        }
    }
    if best.is_finite() { Some(best) } else { None }
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}
fn p95(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((0.95 * s.len() as f64).ceil() as usize).saturating_sub(1).min(s.len() - 1);
    s[idx]
}
