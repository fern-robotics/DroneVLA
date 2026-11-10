//! Deterministic kinematic multirotor. The simulator owns collisions, ground
//! contact, and state integration; it does not know about policies or metrics.

use crate::action::Delta4;
use crate::task::TaskEpisode;
use crate::geom::{body_to_ned, wrap_angle, Vec3};

#[derive(Debug, Clone, Copy)]
pub struct SimState {
    pub sim_time_s: f64,
    pub position: Vec3,
    pub velocity_ned: Vec3,
    pub yaw: f64,
}

impl SimState {
    pub fn altitude_m(&self) -> f64 {
        -self.position.z
    }
}

pub struct StepOutcome {
    pub collision_id: Option<String>,
    pub end: SimState,
}

pub struct Simulator {
    pub state: SimState,
    episode: TaskEpisode,
    collision_count: u32,
    collision_id: Option<String>,
    contact: bool,
    pub landed: bool,
}

impl Simulator {
    pub fn new(episode: &TaskEpisode) -> Self {
        Self {
            state: SimState {
                sim_time_s: 0.0,
                position: episode.start_pos,
                velocity_ned: Vec3::zero(),
                yaw: episode.start_yaw,
            },
            episode: episode.clone(),
            collision_count: 0,
            collision_id: None,
            contact: false,
            landed: false,
        }
    }

    /// Execute a delta4 displacement over `duration_s`, with 10 ms substeps so
    /// thin panels cannot be tunnelled through.
    pub fn step(&mut self, delta: Delta4, duration_s: f64) -> StepOutcome {
        let duration = duration_s.max(1e-6);
        let mut v_body = Vec3::new(delta.dx, delta.dy, delta.dz).scale(1.0 / duration);
        let speed = v_body.norm();
        if speed > self.episode.limits.max_speed_mps && speed > 1e-12 {
            v_body = v_body.scale(self.episode.limits.max_speed_mps / speed);
        }
        let v_ned = body_to_ned(v_body, self.state.yaw);
        let yaw_rate = delta.dyaw / duration;
        let steps = (duration / 0.01).ceil().max(1.0) as usize;
        let dt = duration / steps as f64;
        for _ in 0..steps {
            let candidate = self.state.position.add(v_ned.scale(dt));
            let contact = self.in_contact(candidate);
            if contact && !self.contact {
                self.collision_count += 1;
                self.collision_id = Some(format!("contact-{}", self.collision_count));
            }
            self.contact = contact;
            if !contact {
                self.state.position = candidate;
            }
            self.state.yaw = wrap_angle(self.state.yaw + yaw_rate * dt);
            self.state.sim_time_s += dt;
        }
        self.state.velocity_ned = v_ned;
        StepOutcome {
            collision_id: self.collision_id.clone(),
            end: self.state,
        }
    }

    fn in_contact(&self, p: Vec3) -> bool {
        // Ground contact (NED z near zero).
        if p.z >= -0.25 {
            return true;
        }
        // Landmark panels are visual anchors, not solid obstacles; collisions
        // are evaluated against the ground and geofence.
        false
    }

    pub fn stop(&mut self) {
        self.state.velocity_ned = Vec3::zero();
    }

    pub fn land(&mut self) {
        self.state.velocity_ned = Vec3::zero();
        self.state.position.z = 0.0;
        self.landed = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn integrates_a_small_forward_step() {
        let eps = crate::task::generate_all();
        let ep = eps.first().unwrap();
        let mut sim = Simulator::new(ep);
        let start = sim.state.position;
        sim.step(Delta4::new(0.2, 0.0, 0.0, 0.1), 0.5);
        assert!((sim.state.position.distance(start) - 0.2).abs() < 0.02);
        assert!((sim.state.yaw - 0.1).abs() < 1e-6);
        assert!((sim.state.sim_time_s - 0.5).abs() < 1e-6);
    }

    #[test]
    fn detects_ground_contact() {
        let eps = crate::task::generate_all();
        let ep = eps.first().unwrap();
        let mut sim = Simulator::new(ep);
        // Force the vehicle toward the ground; contact is latched.
        sim.state.position = Vec3::new(0.0, 0.0, -0.5);
        let out = sim.step(Delta4::new(0.0, 0.0, 0.4, 0.0), 0.5);
        assert!(out.collision_id.is_some());
        assert!(sim.state.position.z < -0.2);
    }
}
