//! Fixed intent planner. It converts the current intent and the perceived
//! landmark estimate into a goal pose that the learned motion policy tracks.
//! The planner is identical across M0/M1/M2 and is never trained.

use crate::geom::{ned_to_body, wrap_angle, Vec3};
use crate::instruction::{Direction, IntentSpec, OrbitDir, Side, UpDown};
use crate::sim::SimState;
use crate::task::TaskEpisode;
use crate::tracker::IntentRuntime;

#[derive(Debug, Clone, Copy)]
pub struct Goal {
    pub position: Vec3,
    pub yaw: f64,
}

pub fn plan(
    intent: &IntentSpec,
    rt: &IntentRuntime,
    ep: &TaskEpisode,
    state: SimState,
    landmark: Option<Vec3>,
) -> Goal {
    match intent {
        IntentSpec::Move { direction, distance_m } => {
            let (dx, dy) = direction.body_vector();
            let world = crate::geom::body_to_ned(Vec3::new(dx * distance_m, dy * distance_m, 0.0), rt.start_yaw);
            Goal {
                position: rt.start_pos.add(world),
                yaw: rt.start_yaw,
            }
        }
        IntentSpec::Yaw { direction, degrees } => Goal {
            position: state.position,
            yaw: wrap_angle(rt.start_yaw + direction.sign() * degrees.to_radians()),
        },
        IntentSpec::Altitude { direction, meters } => Goal {
            position: Vec3::new(state.position.x, state.position.y, match direction {
                UpDown::Up => rt.start_pos.z - meters,
                UpDown::Down => rt.start_pos.z + meters,
            }),
            yaw: rt.start_yaw,
        },
        IntentSpec::HoldPosition { .. } => Goal {
            position: rt.start_pos,
            yaw: rt.start_yaw,
        },
        IntentSpec::Approach { distance_m, .. } | IntentSpec::HoldAt { distance_m, .. } => {
            let Some(l) = landmark else {
                return Goal { position: state.position, yaw: state.yaw };
            };
            let dir = horizontal_dir(l, state.position);
            Goal {
                position: Vec3::new(l.x + dir.x * distance_m, l.y + dir.y * distance_m, l.z),
                yaw: face_yaw(state.position, l),
            }
        }
        IntentSpec::Pass { side, clearance_m, .. } => {
            let Some(l) = landmark else {
                return Goal { position: state.position, yaw: state.yaw };
            };
            let travel = horizontal_dir(rt.start_pos, l);
            let right = Vec3::new(-travel.y, travel.x, 0.0);
            let offset = right.scale(side.sign() * clearance_m);
            let pre = l.sub(travel.scale(2.0)).add(offset);
            let post = l.add(travel.scale(3.0)).add(offset);
            let plane = state.position.sub(l).dot(travel);
            // Commit to the crossing leg once the entry waypoint is reached;
            // the threshold is a plane distance, so it does not oscillate.
            let target = if plane < -2.2 { pre } else { post };
            Goal { position: target, yaw: travel.y.atan2(travel.x) }
        }
        IntentSpec::Orbit { direction, radius_m, .. } => {
            let Some(l) = landmark else {
                return Goal { position: state.position, yaw: state.yaw };
            };
            let bearing = (state.position.y - l.y).atan2(state.position.x - l.x);
            let radius = state.position.horizontal_distance(l);
            let step = 0.22 * direction.sign();
            let target_bearing = if radius > radius_m + 1.0 { bearing } else { bearing + step };
            let pos = Vec3::new(
                l.x + radius_m * target_bearing.cos(),
                l.y + radius_m * target_bearing.sin(),
                l.z,
            );
            let tangent = target_bearing + direction.sign() * std::f64::consts::FRAC_PI_2;
            Goal { position: pos, yaw: tangent }
        }
        IntentSpec::Return { .. } => Goal {
            position: rt.start_pos,
            yaw: 0.0,
        },
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

fn face_yaw(from: Vec3, to: Vec3) -> f64 {
    (to.y - from.y).atan2(to.x - from.x)
}

pub trait HorizontalDistance {
    fn horizontal_distance(self, other: Vec3) -> f64;
}
impl HorizontalDistance for Vec3 {
    fn horizontal_distance(self, other: Vec3) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

/// Goal error in the body frame, used by both the policy features and the
/// fixed proportional oracle.
pub fn body_error(state: SimState, goal: Goal) -> (Vec3, f64, f64) {
    let e = goal.position.sub(state.position);
    let body = ned_to_body(e, state.yaw);
    let yaw_err = wrap_angle(goal.yaw - state.yaw);
    (body, yaw_err, e.norm())
}

pub fn _unused(_: Direction, _: OrbitDir, _: Side, _: &TaskEpisode) {}
