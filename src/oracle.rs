//! Privileged intent planner oracle. It validates that every frozen episode is
//! feasible, and its path length is the reference for path efficiency.

use crate::action::Delta4;
use crate::geom::{clamp, wrap_angle, Vec3};
use crate::planner::{self, body_error};
use crate::sim::{SimState, Simulator};
use crate::task::TaskEpisode;
use crate::tracker::MissionTracker;

pub fn proportional_action(state: SimState, goal: planner::Goal) -> Delta4 {
    let (body, yaw_err, _) = body_error(state, goal);
    let mut dx = clamp(body.x * 0.9, -0.32, 0.32);
    let mut dy = clamp(body.y * 0.7, -0.18, 0.18);
    let dz = clamp(body.z * 0.8, -0.16, 0.16);
    let mag = (dx * dx + dy * dy).sqrt();
    if mag > 0.32 {
        dx *= 0.32 / mag;
        dy *= 0.32 / mag;
    }
    Delta4::new(dx, dy, dz, clamp(yaw_err * 0.8, -0.25, 0.25))
}

pub struct OracleResult {
    pub success: bool,
    pub path_length_m: f64,
    pub sim_time_s: f64,
}

pub fn run_oracle(ep: &TaskEpisode) -> OracleResult {
    let mut sim = Simulator::new(ep);
    let mut tracker = MissionTracker::new(ep);
    let mut path = 0.0;
    let mut last = sim.state.position;
    let mut last_idx = usize::MAX;
    let mut memory: Option<Vec3> = None;
    for _ in 0..900 {
        if tracker.is_complete() {
            break;
        }
        if tracker.index != last_idx {
            memory = None;
            last_idx = tracker.index;
        }
        let intent = tracker.current().cloned();
        let rt = tracker.runtime().cloned();
        let Some(intent) = intent else { break };
        let Some(rt) = rt else { break };
        if let Some(desc) = intent.landmark() {
            if let Some(l) = ep.landmark(desc) {
                memory = Some(l.pos);
            }
        }
        let goal = planner::plan(&intent, &rt, ep, sim.state, memory);
        let a = proportional_action(sim.state, goal);
        let out = sim.step(a, 0.1);
        path += last.distance(sim.state.position);
        last = sim.state.position;
        if out.collision_id.is_some() || !ep.limits.contains(sim.state.position) {
            break;
        }
        if tracker.update(sim.state, 0.1) {
            break;
        }
    }
    OracleResult {
        success: tracker.is_complete(),
        path_length_m: path,
        sim_time_s: sim.state.sim_time_s,
    }
}

pub fn validate_episodes(episodes: &mut [TaskEpisode]) -> Vec<String> {
    let mut failures = Vec::new();
    for ep in episodes.iter_mut() {
        let r = run_oracle(ep);
        if !r.success {
            failures.push(format!("{} ({})", ep.id, ep.instruction));
        }
        ep.reference_path_m = r.path_length_m.max(0.1);
    }
    failures
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::generate_all;

    #[test]
    fn oracle_solves_all_intent_episodes() {
        let mut eps = generate_all();
        let failures = validate_episodes(&mut eps);
        assert!(failures.is_empty(), "oracle failures: {failures:#?}");
    }
}
