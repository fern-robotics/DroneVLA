//! Fixed action gate shared by every condition. Policies cannot bypass it.
//! Accepted, clipped, and rejected proposals are all logged.

use crate::action::Delta4;
use crate::task::Limits;
use crate::geom::{body_to_ned, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateDisposition {
    Accepted,
    Clipped,
    Rejected,
}

impl GateDisposition {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Clipped => "clipped",
            Self::Rejected => "rejected",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateDecision {
    pub disposition: GateDisposition,
    pub reasons: Vec<String>,
    pub action: Option<Delta4>,
}

#[derive(Debug, Clone, Copy)]
pub struct GateObservation {
    pub position: Vec3,
    pub yaw: f64,
    pub sim_time_s: f64,
    pub sensor_timestamp_s: Option<f64>,
    pub front_clearance_m: Option<f64>,
}

pub struct SafetyGate {
    pub limits: Limits,
    pub require_depth: bool,
}

impl SafetyGate {
    pub fn new(limits: Limits) -> Self {
        Self {
            limits,
            require_depth: false,
        }
    }

    pub fn check(&self, action: Delta4, obs: &GateObservation, age_s: f64) -> GateDecision {
        let reject = |reason: &str| GateDecision {
            disposition: GateDisposition::Rejected,
            reasons: vec![reason.to_string()],
            action: None,
        };
        if !action.is_finite() {
            return reject("NON_FINITE_ACTION");
        }
        if age_s > self.limits.max_observation_age_s {
            return reject("STALE_OBSERVATION");
        }
        if let Some(ts) = obs.sensor_timestamp_s {
            if (obs.sim_time_s - ts).abs() > self.limits.max_sensor_skew_s {
                return reject("SENSOR_TIME_SKEW");
            }
        }
        if !self.limits.contains(obs.position) {
            return reject("OUTSIDE_GEOFENCE");
        }
        let mut out = action;
        let mut reasons = Vec::new();
        let step = out.step_norm();
        if step > self.limits.max_step_m {
            let s = self.limits.max_step_m / step;
            out.dx *= s;
            out.dy *= s;
            out.dz *= s;
            reasons.push("STEP_CLIPPED".to_string());
        }
        if out.dyaw.abs() > self.limits.max_yaw_step_rad {
            out.dyaw = out.dyaw.signum() * self.limits.max_yaw_step_rad;
            reasons.push("YAW_CLIPPED".to_string());
        }
        let world_delta = body_to_ned(out.translation(), obs.yaw);
        let target = obs.position.add(world_delta);
        if !self.limits.contains(target) {
            return reject("TARGET_OUTSIDE_GEOFENCE");
        }
        // Front depth supports forward motion only; it is not an omnidirectional
        // shield. The depth image is evaluator-only sensor data.
        if out.dx > 0.0 {
            match obs.front_clearance_m {
                None if self.require_depth => return reject("DEPTH_UNAVAILABLE"),
                Some(c) if c < self.limits.min_front_clearance_m + out.dx => {
                    return reject("FRONT_OBSTACLE")
                }
                _ => {}
            }
        }
        GateDecision {
            disposition: if reasons.is_empty() {
                GateDisposition::Accepted
            } else {
                GateDisposition::Clipped
            },
            reasons,
            action: Some(out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs() -> GateObservation {
        GateObservation {
            position: Vec3::new(0.0, 0.0, -10.0),
            yaw: 0.0,
            sim_time_s: 2.0,
            sensor_timestamp_s: Some(2.0),
            front_clearance_m: Some(5.0),
        }
    }

    #[test]
    fn clips_large_steps_and_yaw() {
        let g = SafetyGate::new(Limits::default());
        let d = g.check(Delta4::new(2.0, 0.0, 0.0, 2.0), &obs(), 0.0);
        assert_eq!(d.disposition, GateDisposition::Clipped);
        let a = d.action.unwrap();
        assert!((a.step_norm() - Limits::default().max_step_m).abs() < 1e-9);
        assert!((a.dyaw - Limits::default().max_yaw_step_rad).abs() < 1e-9);
        assert!(d.reasons.contains(&"STEP_CLIPPED".to_string()));
    }

    #[test]
    fn rejects_front_obstacle_and_stale_observation() {
        let g = SafetyGate::new(Limits::default());
        let mut o = obs();
        o.front_clearance_m = Some(0.25);
        let d = g.check(Delta4::new(0.2, 0.0, 0.0, 0.0), &o, 0.0);
        assert_eq!(d.disposition, GateDisposition::Rejected);
        assert_eq!(d.reasons, vec!["FRONT_OBSTACLE"]);

        let d = g.check(Delta4::zero(), &obs(), 99.0);
        assert_eq!(d.reasons, vec!["STALE_OBSERVATION"]);
    }

    #[test]
    fn rejects_target_outside_geofence() {
        let g = SafetyGate::new(Limits::default());
        let mut o = obs();
        o.position = Vec3::new(10.4, 0.0, -10.0);
        let d = g.check(Delta4::new(0.2, 0.0, 0.0, 0.0), &o, 0.0);
        assert_eq!(d.disposition, GateDisposition::Rejected);
        assert_eq!(d.reasons, vec!["TARGET_OUTSIDE_GEOFENCE"]);
    }
}
