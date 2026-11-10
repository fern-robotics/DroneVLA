//! AirSimNH-style intent episodes: frozen manifests over visual landmarks.

use crate::geom::Vec3;
use crate::instruction::{InstructionSpec, IntentSpec, LandmarkDesc, Direction, Side, OrbitDir, UpDown, TurnDir};
use crate::scene::{Color, Marker, Scene, Shape};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Limits {
    pub max_sim_time_s: f64,
    pub max_decisions: u32,
    pub max_step_m: f64,
    pub max_speed_mps: f64,
    pub max_yaw_step_rad: f64,
    pub min_front_clearance_m: f64,
    pub max_observation_age_s: f64,
    pub max_sensor_skew_s: f64,
    pub geofence_min: [f64; 3],
    pub geofence_max: [f64; 3],
    pub terminate_on_collision: bool,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_sim_time_s: 90.0,
            max_decisions: 600,
            max_step_m: 0.40,
            max_speed_mps: 3.0,
            max_yaw_step_rad: 0.30,
            min_front_clearance_m: 0.2,
            max_observation_age_s: 5.0,
            max_sensor_skew_s: 0.5,
            geofence_min: [-2.0, -10.0, -14.0],
            geofence_max: [10.5, 10.0, -2.0],
            terminate_on_collision: true,
        }
    }
}

impl Limits {
    pub fn contains(&self, p: Vec3) -> bool {
        p.x >= self.geofence_min[0]
            && p.x <= self.geofence_max[0]
            && p.y >= self.geofence_min[1]
            && p.y <= self.geofence_max[1]
            && p.z >= self.geofence_min[2]
            && p.z <= self.geofence_max[2]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Landmark {
    pub id: String,
    pub color: Color,
    pub shape: Shape,
    pub pos: Vec3,
    pub size: f64,
}

impl Landmark {
    pub fn desc(&self) -> LandmarkDesc {
        LandmarkDesc::new(self.color, self.shape)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEpisode {
    pub schema_version: String,
    pub id: String,
    pub domain: String,
    pub split: String,
    pub family: String,
    pub ood: bool,
    pub instruction: String,
    pub intents: Vec<IntentSpec>,
    pub start_pos: Vec3,
    pub start_yaw: f64,
    pub landmarks: Vec<Landmark>,
    pub limits: Limits,
    pub reference_path_m: f64,
    pub seed: u64,
}

impl TaskEpisode {
    pub fn scene(&self) -> Scene {
        Scene {
            name: self.id.clone(),
            ground_z: 0.0,
            markers: self
                .landmarks
                .iter()
                .map(|l| Marker {
                    id: l.id.clone(),
                    color: l.color,
                    shape: l.shape,
                    pos: l.pos,
                    size: l.size,
                })
                .collect(),
        }
    }
    pub fn landmark(&self, desc: LandmarkDesc) -> Option<&Landmark> {
        self.landmarks
            .iter()
            .find(|l| l.color == desc.color && l.shape == desc.shape)
    }
    pub fn instruction_spec(&self) -> InstructionSpec {
        InstructionSpec {
            raw: self.instruction.clone(),
            intents: self.intents.clone(),
        }
    }
    pub fn validate(&self) -> Result<()> {
        if self.intents.is_empty() {
            anyhow::bail!("episode {} has no intents", self.id);
        }
        if !self.limits.contains(self.start_pos) {
            anyhow::bail!("start outside geofence");
        }
        for intent in &self.intents {
            if let Some(desc) = intent.landmark() {
                if self.landmark(desc).is_none() {
                    anyhow::bail!("missing landmark {} in {}", desc.name(), self.id);
                }
            }
        }
        let parsed = InstructionSpec::parse(&self.instruction)
            .map_err(|e| anyhow::anyhow!("instruction parse: {e}"))?;
        if parsed.intents != self.intents {
            anyhow::bail!("instruction and intent list disagree in {}", self.id);
        }
        Ok(())
    }
}

// Landmark appearance set. Unique descriptors are used for instructed targets;
// the second blue circle is a distractor.
const APPEARANCES: [(&str, Color, Shape); 6] = [
    ("A", Color::Blue, Shape::Circle),
    ("B", Color::Yellow, Shape::Triangle),
    ("C", Color::Red, Shape::Circle),
    ("D", Color::Green, Shape::Cross),
    ("E", Color::Blue, Shape::Circle),
    ("F", Color::Blue, Shape::Square),
];
const BASE_Y: [f64; 6] = [-6.0, -3.6, -1.2, 1.2, 3.6, 6.0];
const PERM_A: [usize; 6] = [0, 1, 2, 3, 4, 5];
const PERM_B: [usize; 6] = [3, 0, 4, 1, 5, 2];
const PERM_C: [usize; 6] = [0, 2, 5, 1, 3, 4];
const X_OFF_B: [f64; 6] = [-0.3, 0.2, -0.1, 0.3, 0.0, -0.2];
const X_OFF_C: [f64; 6] = [0.2, -0.3, 0.3, -0.1, 0.1, -0.2];

fn landmarks(variant: &str) -> Vec<Landmark> {
    let (perm, xoff) = match variant {
        "a" => (PERM_A, [0.0; 6]),
        "b" => (PERM_B, X_OFF_B),
        _ => (PERM_C, X_OFF_C),
    };
    APPEARANCES
        .iter()
        .enumerate()
        .map(|(i, (id, color, shape))| Landmark {
            id: id.to_string(),
            color: *color,
            shape: *shape,
            pos: Vec3::new(6.0 + xoff[i], BASE_Y[perm[i]], -10.0),
            size: 1.5,
        })
        .collect()
}

fn desc(color: Color, shape: Shape) -> LandmarkDesc {
    LandmarkDesc::new(color, shape)
}

fn unique_landmark(landmarks: &[Landmark], desc: LandmarkDesc) -> Option<&Landmark> {
    let hits: Vec<&Landmark> = landmarks
        .iter()
        .filter(|l| l.color == desc.color && l.shape == desc.shape)
        .collect();
    if hits.len() == 1 {
        Some(hits[0])
    } else {
        None
    }
}

/// Intent templates by role.
fn primitive_intents(instance: usize) -> Vec<IntentSpec> {
    let cases = [
        vec![IntentSpec::Move { direction: Direction::Forward, distance_m: 3.0 }],
        vec![IntentSpec::Move { direction: Direction::Backward, distance_m: 2.0 }],
        vec![IntentSpec::Move { direction: Direction::Left, distance_m: 2.0 }],
        vec![IntentSpec::Move { direction: Direction::Right, distance_m: 2.0 }],
        vec![IntentSpec::Yaw { direction: TurnDir::Left, degrees: 90.0 }],
        vec![IntentSpec::Yaw { direction: TurnDir::Right, degrees: 45.0 }],
        vec![IntentSpec::Altitude { direction: UpDown::Up, meters: 2.0 }],
        vec![IntentSpec::Altitude { direction: UpDown::Down, meters: 2.0 }],
        vec![IntentSpec::HoldPosition { duration_s: 2.0, tolerance_m: 0.8 }],
        vec![IntentSpec::Move { direction: Direction::Forward, distance_m: 4.0 }],
    ];
    cases[instance % cases.len()].clone()
}

fn interaction_intents(landmarks: &[Landmark], instance: usize) -> Vec<IntentSpec> {
    let red = unique_landmark(landmarks, desc(Color::Red, Shape::Circle)).unwrap().pos;
    let _ = red;
    let target = match instance % 4 {
        0 => desc(Color::Red, Shape::Circle),
        1 => desc(Color::Blue, Shape::Square),
        2 => desc(Color::Green, Shape::Cross),
        _ => desc(Color::Yellow, Shape::Triangle),
    };
    match instance % 8 {
        0 => vec![IntentSpec::Approach { landmark: target, distance_m: 4.0, tolerance_m: 0.9 }],
        1 => vec![IntentSpec::Approach { landmark: target, distance_m: 4.0, tolerance_m: 0.9 }],
        2 => vec![IntentSpec::Pass { landmark: target, side: Side::Left, clearance_m: 1.2 }],
        3 => vec![IntentSpec::Pass { landmark: target, side: Side::Right, clearance_m: 1.2 }],
        4 => vec![IntentSpec::Orbit { landmark: target, direction: OrbitDir::Clockwise, min_degrees: 180.0, radius_m: 3.5 }],
        5 => vec![IntentSpec::Orbit { landmark: target, direction: OrbitDir::CounterClockwise, min_degrees: 180.0, radius_m: 3.5 }],
        6 => vec![IntentSpec::HoldAt { landmark: target, distance_m: 4.0, tolerance_m: 0.9, duration_s: 2.0 }],
        _ => vec![IntentSpec::HoldAt { landmark: target, distance_m: 3.5, tolerance_m: 0.9, duration_s: 2.5 }],
    }
}

fn composition_intents(landmarks: &[Landmark], template: usize, ood: bool) -> Vec<IntentSpec> {
    let red = desc(Color::Red, Shape::Circle);
    let square = desc(Color::Blue, Shape::Square);
    let cross = desc(Color::Green, Shape::Cross);
    let triangle = desc(Color::Yellow, Shape::Triangle);
    let _ = landmarks;
    if !ood {
        match template % 4 {
            0 => vec![
                IntentSpec::Approach { landmark: red, distance_m: 4.0, tolerance_m: 0.9 },
                IntentSpec::Return { tolerance_m: 0.8 },
            ],
            1 => vec![
                IntentSpec::Pass { landmark: square, side: Side::Left, clearance_m: 1.2 },
                IntentSpec::Return { tolerance_m: 0.8 },
            ],
            2 => vec![
                IntentSpec::Orbit { landmark: cross, direction: OrbitDir::Clockwise, min_degrees: 180.0, radius_m: 3.5 },
                IntentSpec::Return { tolerance_m: 0.8 },
            ],
            _ => vec![
                IntentSpec::Move { direction: Direction::Forward, distance_m: 3.0 },
                IntentSpec::Yaw { direction: TurnDir::Left, degrees: 90.0 },
            ],
        }
    } else {
        match template % 4 {
            0 => vec![
                IntentSpec::Approach { landmark: red, distance_m: 4.0, tolerance_m: 0.9 },
                IntentSpec::Pass { landmark: red, side: Side::Left, clearance_m: 1.2 },
                IntentSpec::Return { tolerance_m: 0.8 },
            ],
            1 => vec![
                IntentSpec::Pass { landmark: square, side: Side::Right, clearance_m: 1.2 },
                IntentSpec::Orbit { landmark: cross, direction: OrbitDir::CounterClockwise, min_degrees: 180.0, radius_m: 3.5 },
            ],
            2 => vec![
                IntentSpec::Orbit { landmark: cross, direction: OrbitDir::CounterClockwise, min_degrees: 180.0, radius_m: 3.5 },
                IntentSpec::HoldAt { landmark: triangle, distance_m: 4.0, tolerance_m: 0.9, duration_s: 2.0 },
                IntentSpec::Return { tolerance_m: 0.8 },
            ],
            _ => vec![
                IntentSpec::Move { direction: Direction::Forward, distance_m: 2.0 },
                IntentSpec::Pass { landmark: square, side: Side::Left, clearance_m: 1.2 },
                IntentSpec::Return { tolerance_m: 0.8 },
            ],
        }
    }
}

fn make_episode(
    variant: &str,
    split: &str,
    family: &str,
    instance: usize,
    ood: bool,
    intents: Vec<IntentSpec>,
) -> TaskEpisode {
    let landmarks = landmarks(variant);
    // Align the start with the first instructed landmark when the intent refers
    // to one, so that pass/orbit geometry is unambiguous.
    let first_lm = intents.iter().find_map(|i| i.landmark());
    let y = first_lm
        .and_then(|d| unique_landmark(&landmarks, d))
        .map(|l| l.pos.y)
        .unwrap_or(0.0);
    let start_pos = Vec3::new(0.0, y, -10.0);
    let instruction = InstructionSpec::render(&intents);
    let id = format!("airsimnh-{family}-{variant}-{instance:02}");
    let seed = fnv(&id);
    TaskEpisode {
        schema_version: "2.0".into(),
        id,
        domain: "AirSimNH".into(),
        split: split.into(),
        family: family.into(),
        ood,
        instruction,
        intents,
        start_pos,
        start_yaw: 0.0,
        landmarks,
        limits: Limits::default(),
        reference_path_m: 0.0,
        seed,
    }
}

fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn generate_all() -> Vec<TaskEpisode> {
    let mut out = Vec::new();
    for (variant, split) in [("a", "train"), ("b", "dev"), ("c", "test")] {
        let lms = landmarks(variant);
        // Primitive motion: 10 per split.
        for i in 0..10 {
            out.push(make_episode(variant, split, "primitive", i, false, primitive_intents(i)));
        }
        // Landmark interaction: 12 per split.
        for i in 0..12 {
            out.push(make_episode(variant, split, "interaction", i, false, interaction_intents(&lms, i)));
        }
        // Composition: 8 train/dev, 16 test (8 ID, 8 OOD).
        let n = if split == "test" { 16 } else { 8 };
        for i in 0..n {
            let ood = split == "test" && i >= 8;
            let template = i % 8;
            out.push(make_episode(
                variant,
                split,
                "composition",
                i,
                ood,
                composition_intents(&lms, template, ood),
            ));
        }
    }
    out
}

pub fn write_suite(dir: &std::path::Path, episodes: &[TaskEpisode]) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    for ep in episodes {
        ep.validate()?;
        std::fs::write(
            dir.join(format!("{}.json", ep.id)),
            serde_json::to_string_pretty(ep)?,
        )?;
    }
    let suite = serde_json::json!({
        "schema_version": "2.0",
        "suite_id": "dronevl-intent-airsimnh-v2",
        "episodes": episodes.iter().map(|e| serde_json::json!({
            "episode": format!("{}.json", e.id),
            "id": e.id,
            "family": e.family,
            "variant": e.domain,
            "split": e.split,
            "ood": e.ood,
        })).collect::<Vec<_>>(),
    });
    std::fs::write(dir.join("suite.json"), serde_json::to_string_pretty(&suite)?)?;
    Ok(())
}

pub fn load_suite(dir: &std::path::Path) -> Result<Vec<TaskEpisode>> {
    let text = std::fs::read_to_string(dir.join("suite.json"))?;
    let suite: serde_json::Value = serde_json::from_str(&text)?;
    let mut out = Vec::new();
    for entry in suite["episodes"].as_array().cloned().unwrap_or_default() {
        let file = dir.join(entry["episode"].as_str().unwrap_or_default());
        let ep: TaskEpisode = serde_json::from_str(&std::fs::read_to_string(file)?)?;
        ep.validate()?;
        out.push(ep);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_suite_is_valid_and_balanced() {
        let eps = generate_all();
        assert_eq!(eps.len(), 98);
        for e in &eps {
            e.validate().unwrap();
        }
        let test: Vec<_> = eps.iter().filter(|e| e.split == "test").collect();
        assert_eq!(test.len(), 38);
        assert_eq!(test.iter().filter(|e| e.ood).count(), 8);
        let ids: HashSet<_> = eps.iter().map(|e| e.id.clone()).collect();
        assert_eq!(ids.len(), eps.len());
    }
}
