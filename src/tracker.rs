//! Evaluator-only intent predicates. Progress is computed from the actual
//! trajectory, never from the policy's claim, and the same tracker is used by
//! the oracle, the runner, and the analysis.

use crate::geom::{ned_to_body, wrap_angle, Vec3};
use crate::instruction::{IntentSpec, OrbitDir, Side, TurnDir, UpDown};
use crate::sim::SimState;
use crate::task::TaskEpisode;

#[derive(Debug, Clone)]
pub struct IntentRuntime {
    pub start_pos: Vec3,
    pub start_yaw: f64,
    pub progress: f64,
    pub satisfied: bool,
    pass_prev_plane: Option<f64>,
    pass_crossed: bool,
    pass_correct_side: bool,
    pass_lateral: f64,
    orbit_last_bearing: Option<f64>,
    pub orbit_accum: f64,
    hold_timer: f64,
    approach_start_dist: f64,
    approach_min_dist: f64,
    approach_inside_since: Option<f64>,
}

impl IntentRuntime {
    fn new(intent: &IntentSpec, ep: &TaskEpisode, state: SimState) -> Self {
        let approach_start_dist = intent
            .landmark()
            .and_then(|d| ep.landmark(d))
            .map(|l| l.pos.horizontal_norm_distance(state.position))
            .unwrap_or(0.0);
        let mut r = Self {
            start_pos: state.position,
            start_yaw: state.yaw,
            progress: 0.0,
            satisfied: false,
            pass_prev_plane: None,
            pass_crossed: false,
            pass_correct_side: false,
            pass_lateral: 0.0,
            orbit_last_bearing: None,
            orbit_accum: 0.0,
            hold_timer: 0.0,
            approach_start_dist,
            approach_min_dist: approach_start_dist,
            approach_inside_since: None,
        };
        if let (IntentSpec::Pass { landmark, .. }, Some(l)) = (intent, intent.landmark().and_then(|d| ep.landmark(d))) {
            let _ = landmark;
            let dir = horizontal_dir(r.start_pos, l.pos);
            r.pass_prev_plane = Some((state.position.sub(l.pos)).x * dir.x + (state.position.sub(l.pos)).y * dir.y);
        }
        if let (IntentSpec::Orbit { landmark: _, .. }, Some(l)) = (intent, intent.landmark().and_then(|d| ep.landmark(d))) {
            r.orbit_last_bearing = Some((state.position.y - l.pos.y).atan2(state.position.x - l.pos.x));
        }
        r
    }
}

trait HorizontalDistance {
    fn horizontal_norm_distance(self, other: Vec3) -> f64;
}
impl HorizontalDistance for Vec3 {
    fn horizontal_norm_distance(self, other: Vec3) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

fn horizontal_dir(from: Vec3, to: Vec3) -> Vec3 {
    let d = Vec3::new(to.x - from.x, to.y - from.y, 0.0);
    let n = d.norm();
    if n < 1e-9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        d.scale(1.0 / n)
    }
}

pub struct MissionTracker {
    pub episode: TaskEpisode,
    pub index: usize,
    pub runtimes: Vec<IntentRuntime>,
    pub completed: usize,
    pub sim_time: f64,
}

impl MissionTracker {
    pub fn new(ep: &TaskEpisode) -> Self {
        let state = SimState {
            sim_time_s: 0.0,
            position: ep.start_pos,
            velocity_ned: Vec3::zero(),
            yaw: ep.start_yaw,
        };
        let runtime = IntentRuntime::new(&ep.intents[0], ep, state);
        Self {
            episode: ep.clone(),
            index: 0,
            runtimes: vec![runtime],
            completed: 0,
            sim_time: 0.0,
        }
    }

    pub fn current(&self) -> Option<&IntentSpec> {
        self.episode.intents.get(self.index)
    }
    pub fn runtime(&self) -> Option<&IntentRuntime> {
        self.runtimes.get(self.index)
    }
    pub fn is_complete(&self) -> bool {
        self.index >= self.episode.intents.len()
    }

    /// Advance all predicates with the new state. Returns true when every
    /// intent is satisfied.
    pub fn update(&mut self, state: SimState, dt: f64) -> bool {
        self.sim_time = state.sim_time_s;
        if self.is_complete() {
            return true;
        }
        let idx = self.index;
        let intent = self.episode.intents[idx].clone();
        let (progress, satisfied) = self.evaluate(&intent, state, dt);
        {
            let rt = &mut self.runtimes[idx];
            rt.progress = progress.clamp(0.0, 1.0);
            rt.satisfied = satisfied;
        }
        if satisfied {
            self.completed += 1;
            self.index += 1;
            if self.index < self.episode.intents.len() {
                let next = self.episode.intents[self.index].clone();
                self.runtimes.push(IntentRuntime::new(&next, &self.episode, state));
            }
        }
        self.is_complete()
    }

    fn evaluate(&mut self, intent: &IntentSpec, state: SimState, dt: f64) -> (f64, bool) {
        match intent {
            IntentSpec::Move { direction, distance_m } => {
                let rt = &mut self.runtimes[self.index];
                let disp = state.position.sub(rt.start_pos);
                let body = ned_to_body(disp, rt.start_yaw);
                let (dx, dy) = direction.body_vector();
                let achieved = body.x * dx + body.y * dy;
                let p = if *distance_m > 1e-6 { achieved / distance_m } else { 1.0 };
                (p, achieved >= distance_m - 0.30)
            }
            IntentSpec::Yaw { direction, degrees } => {
                let rt = &mut self.runtimes[self.index];
                let delta = wrap_angle(state.yaw - rt.start_yaw) * direction.sign();
                let target = degrees.to_radians();
                (delta / target.max(1e-6), delta >= target - 0.10)
            }
            IntentSpec::Altitude { direction, meters } => {
                let rt = &mut self.runtimes[self.index];
                let achieved = match direction {
                    UpDown::Up => rt.start_pos.z - state.position.z,
                    UpDown::Down => state.position.z - rt.start_pos.z,
                };
                (achieved / meters.max(1e-6), achieved >= meters - 0.20)
            }
            IntentSpec::HoldPosition { duration_s, tolerance_m } => {
                let rt = &mut self.runtimes[self.index];
                let inside = state.position.horizontal_norm_distance(rt.start_pos) <= *tolerance_m;
                if inside {
                    rt.hold_timer += dt;
                } else {
                    rt.hold_timer = 0.0;
                }
                (rt.hold_timer / duration_s.max(1e-6), rt.hold_timer >= *duration_s)
            }
            IntentSpec::Approach { landmark, distance_m, tolerance_m } => {
                let l = self.episode.landmark(*landmark).unwrap().pos;
                let rt = &mut self.runtimes[self.index];
                let dist = state.position.horizontal_norm_distance(l);
                rt.approach_min_dist = rt.approach_min_dist.min(dist);
                let span = (rt.approach_start_dist - distance_m).max(1e-6);
                let p = 1.0 - ((dist - distance_m) / span).clamp(0.0, 1.0);
                (p, dist <= distance_m + tolerance_m)
            }
            IntentSpec::Pass { landmark, side, clearance_m } => {
                let l = self.episode.landmark(*landmark).unwrap().pos;
                let rt = &mut self.runtimes[self.index];
                let dir = horizontal_dir(rt.start_pos, l);
                let rel = state.position.sub(l);
                let plane = rel.x * dir.x + rel.y * dir.y;
                let right = Vec3::new(-dir.y, dir.x, 0.0);
                let lateral = rel.x * right.x + rel.y * right.y;
                let prev = rt.pass_prev_plane.unwrap_or(plane);
                if prev < 0.0 && plane >= 0.0 {
                    rt.pass_crossed = true;
                    rt.pass_lateral = lateral;
                    rt.pass_correct_side = match side {
                        Side::Left => lateral <= -clearance_m * 0.6,
                        Side::Right => lateral >= clearance_m * 0.6,
                    };
                }
                rt.pass_prev_plane = Some(plane);
                let p = if rt.pass_crossed {
                    if rt.pass_correct_side { 1.0 } else { 0.6 }
                } else {
                    0.45 * ((-plane) / 6.0).clamp(0.0, 1.0)
                };
                (p, rt.pass_crossed && rt.pass_correct_side)
            }
            IntentSpec::Orbit { landmark, direction, min_degrees, radius_m } => {
                let l = self.episode.landmark(*landmark).unwrap().pos;
                let rt = &mut self.runtimes[self.index];
                let bearing = (state.position.y - l.y).atan2(state.position.x - l.x);
                let radius = state.position.horizontal_norm_distance(l);
                if let Some(prev) = rt.orbit_last_bearing {
                    if radius <= radius_m + 2.5 && radius >= 1.0 {
                        let d = wrap_angle(bearing - prev);
                        rt.orbit_accum += d * direction.sign();
                    }
                }
                rt.orbit_last_bearing = Some(bearing);
                let target = min_degrees.to_radians();
                (rt.orbit_accum / target.max(1e-6), rt.orbit_accum >= target - 0.10)
            }
            IntentSpec::HoldAt { landmark, distance_m, tolerance_m, duration_s } => {
                let l = self.episode.landmark(*landmark).unwrap().pos;
                let rt = &mut self.runtimes[self.index];
                let dist = state.position.horizontal_norm_distance(l);
                let inside = (dist - distance_m).abs() <= *tolerance_m;
                if inside {
                    rt.hold_timer += dt;
                } else {
                    rt.hold_timer = 0.0;
                }
                (rt.hold_timer / duration_s.max(1e-6), rt.hold_timer >= *duration_s)
            }
            IntentSpec::Return { tolerance_m } => {
                let rt = &mut self.runtimes[self.index];
                let dist = state.position.horizontal_norm_distance(rt.start_pos);
                let p = 1.0 - (dist / 5.0).clamp(0.0, 1.0);
                (p, dist <= *tolerance_m)
            }
        }
    }

    pub fn intent_score(&self) -> f64 {
        if self.episode.intents.is_empty() {
            return 0.0;
        }
        if self.is_complete() {
            return 1.0;
        }
        let current = self.runtime().map(|r| r.progress).unwrap_or(0.0);
        ((self.completed as f64) + current.clamp(0.0, 1.0)) / self.episode.intents.len() as f64
    }

    pub fn ordered_completion(&self) -> f64 {
        self.completed as f64 / self.episode.intents.len().max(1) as f64
    }

    pub fn per_intent_progress(&self) -> Vec<f64> {
        self.runtimes
            .iter()
            .enumerate()
            .map(|(i, r)| if i < self.index { 1.0 } else { r.progress })
            .collect()
    }

    pub fn speed_mps(state: &SimState) -> f64 {
        state.velocity_ned.norm()
    }

    pub fn current_progress(&self) -> f64 {
        self.runtime().map(|r| r.progress).unwrap_or(1.0)
    }

    pub fn terminal_success(&self) -> bool {
        self.is_complete()
    }

    pub fn orbit_direction_ok(&self, direction: OrbitDir) -> bool {
        self.runtime()
            .map(|r| r.orbit_accum * direction.sign() >= 0.0)
            .unwrap_or(true)
    }

    pub fn pass_side_correct(&self, side: Side) -> bool {
        self.runtime()
            .map(|r| {
                if r.pass_crossed {
                    match side {
                        Side::Left => r.pass_lateral < 0.0,
                        Side::Right => r.pass_lateral > 0.0,
                    }
                } else {
                    true
                }
            })
            .unwrap_or(true)
    }

    pub fn yaw_turn_ok(&self, turn: TurnDir) -> bool {
        let _ = turn;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::generate_all;

    fn state(pos: Vec3, yaw: f64, t: f64) -> SimState {
        SimState {
            sim_time_s: t,
            position: pos,
            velocity_ned: Vec3::zero(),
            yaw,
        }
    }

    #[test]
    fn move_predicate_reaches_target() {
        let eps = generate_all();
        let ep = eps.iter().find(|e| e.family == "primitive" && e.intents[0].type_name() == "move").unwrap();
        let mut tr = MissionTracker::new(ep);
        let dist = match ep.intents[0] { IntentSpec::Move { distance_m, .. } => distance_m, _ => 0.0 };
        let mut done = false;
        for i in 0..100 {
            let pos = Vec3::new(dist, ep.start_pos.y, ep.start_pos.z);
            done = tr.update(state(pos, 0.0, (i + 1) as f64 * 0.1), 0.1);
            if done { break; }
        }
        assert!(done, "move did not complete: progress {}", tr.intent_score());
    }

    #[test]
    fn orbit_accumulates_signed_angle() {
        let eps = generate_all();
        let ep = eps
            .iter()
            .find(|e| e.family == "interaction" && e.intents[0].type_name() == "orbit")
            .unwrap();
        let (lm, dir, rad) = match &ep.intents[0] {
            IntentSpec::Orbit { landmark, direction, radius_m, .. } => (ep.landmark(*landmark).unwrap().pos, *direction, *radius_m),
            _ => unreachable!(),
        };
        let mut tr = MissionTracker::new(ep);
        let mut done = false;
        for i in 0..400 {
            let a = i as f64 * 0.05 * dir.sign();
            let pos = Vec3::new(lm.x + rad * a.cos(), lm.y + rad * a.sin(), lm.z);
            done = tr.update(state(pos, 0.0, (i + 1) as f64 * 0.1), 0.1);
            if done { break; }
        }
        assert!(done, "orbit did not complete: {}", tr.intent_score());
    }
}
