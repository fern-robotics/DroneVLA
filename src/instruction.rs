//! Instruction grammar for the revised flight-intent experiment.
//!
//! Families: primitive motion (translate / yaw / altitude / hold position),
//! landmark interaction (approach / pass-side / orbit / hold-at-distance), and
//! compositional sequences of two or more intents.

use crate::scene::{Color, Shape};
use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LandmarkDesc {
    pub color: Color,
    pub shape: Shape,
}

impl LandmarkDesc {
    pub fn new(color: Color, shape: Shape) -> Self {
        Self { color, shape }
    }
    pub fn name(&self) -> String {
        format!("{} {}", self.color, self.shape)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    Forward,
    Backward,
    Left,
    Right,
}

impl Direction {
    /// Unit vector in the yaw-aligned body frame (forward, right, down).
    pub fn body_vector(self) -> (f64, f64) {
        match self {
            Self::Forward => (1.0, 0.0),
            Self::Backward => (-1.0, 0.0),
            Self::Left => (0.0, -1.0),
            Self::Right => (0.0, 1.0),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Backward => "backward",
            Self::Left => "left",
            Self::Right => "right",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnDir {
    Left,
    Right,
}

impl TurnDir {
    pub fn sign(self) -> f64 {
        match self {
            Self::Left => -1.0,
            Self::Right => 1.0,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Right => "right",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpDown {
    Up,
    Down,
}

impl UpDown {
    pub fn sign(self) -> f64 {
        match self {
            Self::Up => 1.0,
            Self::Down => -1.0,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Up => "up",
            Self::Down => "down",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    Left,
    Right,
}

impl Side {
    /// +1 is right of the travel direction (NED y).
    pub fn sign(self) -> f64 {
        match self {
            Self::Left => -1.0,
            Self::Right => 1.0,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Right => "right",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrbitDir {
    Clockwise,
    CounterClockwise,
}

impl OrbitDir {
    pub fn sign(self) -> f64 {
        match self {
            Self::Clockwise => 1.0,
            Self::CounterClockwise => -1.0,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Clockwise => "clockwise",
            Self::CounterClockwise => "counterclockwise",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IntentSpec {
    Move {
        direction: Direction,
        distance_m: f64,
    },
    Yaw {
        direction: TurnDir,
        degrees: f64,
    },
    Altitude {
        direction: UpDown,
        meters: f64,
    },
    HoldPosition {
        duration_s: f64,
        tolerance_m: f64,
    },
    Approach {
        landmark: LandmarkDesc,
        distance_m: f64,
        tolerance_m: f64,
    },
    Pass {
        landmark: LandmarkDesc,
        side: Side,
        clearance_m: f64,
    },
    Orbit {
        landmark: LandmarkDesc,
        direction: OrbitDir,
        min_degrees: f64,
        radius_m: f64,
    },
    HoldAt {
        landmark: LandmarkDesc,
        distance_m: f64,
        tolerance_m: f64,
        duration_s: f64,
    },
    Return {
        tolerance_m: f64,
    },
}

impl IntentSpec {
    pub fn landmark(&self) -> Option<LandmarkDesc> {
        match self {
            Self::Approach { landmark, .. }
            | Self::Pass { landmark, .. }
            | Self::Orbit { landmark, .. }
            | Self::HoldAt { landmark, .. } => Some(*landmark),
            _ => None,
        }
    }
    pub fn type_index(&self) -> usize {
        match self {
            Self::Move { .. } => 0,
            Self::Yaw { .. } => 1,
            Self::Altitude { .. } => 2,
            Self::HoldPosition { .. } => 3,
            Self::Approach { .. } => 4,
            Self::Pass { .. } => 5,
            Self::Orbit { .. } => 6,
            Self::HoldAt { .. } => 7,
            Self::Return { .. } => 8,
        }
    }
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Move { .. } => "move",
            Self::Yaw { .. } => "yaw",
            Self::Altitude { .. } => "altitude",
            Self::HoldPosition { .. } => "hold_position",
            Self::Approach { .. } => "approach",
            Self::Pass { .. } => "pass",
            Self::Orbit { .. } => "orbit",
            Self::HoldAt { .. } => "hold_at",
            Self::Return { .. } => "return",
        }
    }
    pub fn magnitude(&self) -> f64 {
        match self {
            Self::Move { distance_m, .. } => *distance_m,
            Self::Yaw { degrees, .. } => *degrees,
            Self::Altitude { meters, .. } => *meters,
            Self::HoldPosition { duration_s, .. } => *duration_s,
            Self::Approach { distance_m, .. } => *distance_m,
            Self::Pass { clearance_m, .. } => *clearance_m,
            Self::Orbit { min_degrees, .. } => *min_degrees,
            Self::HoldAt { distance_m, .. } => *distance_m,
            Self::Return { .. } => 0.0,
        }
    }
    pub fn side_sign(&self) -> f64 {
        match self {
            Self::Pass { side, .. } => side.sign(),
            Self::Orbit { direction, .. } => direction.sign(),
            _ => 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstructionSpec {
    pub raw: String,
    pub intents: Vec<IntentSpec>,
}

impl InstructionSpec {
    pub fn render(intents: &[IntentSpec]) -> String {
        let parts: Vec<String> = intents.iter().map(render_intent).collect();
        format!("{}.", parts.join(", then "))
    }

    pub fn parse(raw: &str) -> Result<Self, String> {
        let lower = raw.to_lowercase();
        let lower = lower.trim().trim_end_matches('.').to_string();
        let clauses: Vec<&str> = lower.split(", then ").collect();
        if clauses.is_empty() {
            return Err("empty instruction".into());
        }
        let mut intents = Vec::new();
        for clause in clauses {
            intents.push(parse_clause(clause.trim())?);
        }
        Ok(Self {
            raw: raw.to_string(),
            intents,
        })
    }
}

fn render_intent(intent: &IntentSpec) -> String {
    match intent {
        IntentSpec::Move { direction, distance_m } => {
            format!("Move {} {:.1} metres", direction.name(), distance_m)
        }
        IntentSpec::Yaw { direction, degrees } => {
            format!("Turn {} {:.0} degrees", direction.name(), degrees)
        }
        IntentSpec::Altitude { direction, meters } => match direction {
            UpDown::Up => format!("Climb {:.1} metres", meters),
            UpDown::Down => format!("Descend {:.1} metres", meters),
        },
        IntentSpec::HoldPosition { duration_s, .. } => {
            format!("Hold position for {:.1} seconds", duration_s)
        }
        IntentSpec::Approach { landmark, distance_m, .. } => format!(
            "Approach the {} to within {:.1} metres",
            landmark.name(),
            distance_m
        ),
        IntentSpec::Pass { landmark, side, .. } => {
            format!("Pass the {} on the {}", landmark.name(), side.name())
        }
        IntentSpec::Orbit { landmark, direction, min_degrees, radius_m } => format!(
            "Orbit the {} {} for {:.0} degrees at {:.1} metres",
            landmark.name(),
            direction.name(),
            min_degrees,
            radius_m
        ),
        IntentSpec::HoldAt { landmark, distance_m, duration_s, .. } => format!(
            "Hold {:.1} metres from the {} for {:.1} seconds",
            distance_m,
            landmark.name(),
            duration_s
        ),
        IntentSpec::Return { .. } => "Return to the start".to_string(),
    }
}

fn parse_clause(clause: &str) -> Result<IntentSpec, String> {
    let c = clause
        .trim()
        .trim_start_matches("please ")
        .trim_start_matches("now ")
        .trim();
    if c.starts_with("move forward") || c.starts_with("move backward") || c.starts_with("move left") || c.starts_with("move right") {
        let (direction, rest) = if c.starts_with("move forward") {
            (Direction::Forward, c.trim_start_matches("move forward"))
        } else if c.starts_with("move backward") {
            (Direction::Backward, c.trim_start_matches("move backward"))
        } else if c.starts_with("move left") {
            (Direction::Left, c.trim_start_matches("move left"))
        } else {
            (Direction::Right, c.trim_start_matches("move right"))
        };
        let distance = number_before(rest, "metre").or_else(|| number_before(rest, "meter"))
            .ok_or_else(|| format!("missing distance: {clause}"))?;
        return Ok(IntentSpec::Move { direction, distance_m: distance });
    }
    if c.starts_with("climb") || c.starts_with("ascend") {
        let rest = c.trim_start_matches("climb").trim_start_matches("ascend");
        let meters = number_before(rest, "metre").or_else(|| number_before(rest, "meter"))
            .ok_or_else(|| format!("missing altitude: {clause}"))?;
        return Ok(IntentSpec::Altitude { direction: UpDown::Up, meters });
    }
    if c.starts_with("descend") {
        let rest = c.trim_start_matches("descend");
        let meters = number_before(rest, "metre").or_else(|| number_before(rest, "meter"))
            .ok_or_else(|| format!("missing altitude: {clause}"))?;
        return Ok(IntentSpec::Altitude { direction: UpDown::Down, meters });
    }
    if c.starts_with("turn left") || c.starts_with("turn right") {
        let (direction, rest) = if c.starts_with("turn left") {
            (TurnDir::Left, c.trim_start_matches("turn left"))
        } else {
            (TurnDir::Right, c.trim_start_matches("turn right"))
        };
        let degrees = number_before(rest, "degree")
            .ok_or_else(|| format!("missing turn angle: {clause}"))?;
        return Ok(IntentSpec::Yaw { direction, degrees });
    }
    if c.starts_with("hold position") {
        let duration = number_before(c, "second")
            .ok_or_else(|| format!("missing hold duration: {clause}"))?;
        return Ok(IntentSpec::HoldPosition { duration_s: duration, tolerance_m: 0.8 });
    }
    if c.starts_with("approach the ") || c.starts_with("approach ") {
        let rest = c.trim_start_matches("approach the ").trim_start_matches("approach ");
        let landmark = parse_landmark_prefix(rest)?;
        let distance = number_after(rest, "within ")
            .ok_or_else(|| format!("missing approach distance: {clause}"))?;
        return Ok(IntentSpec::Approach { landmark, distance_m: distance, tolerance_m: 0.9 });
    }
    if c.starts_with("pass the ") {
        let rest = c.trim_start_matches("pass the ");
        let landmark = parse_landmark_prefix(rest)?;
        let side = if c.contains(" on the left") { Side::Left } else if c.contains(" on the right") { Side::Right } else { return Err(format!("missing pass side: {clause}")); };
        return Ok(IntentSpec::Pass { landmark, side, clearance_m: 1.2 });
    }
    if c.starts_with("orbit the ") {
        let rest = c.trim_start_matches("orbit the ");
        let landmark = parse_landmark_prefix(rest)?;
        let direction = if c.contains("counterclockwise") { OrbitDir::CounterClockwise } else if c.contains("clockwise") { OrbitDir::Clockwise } else { return Err(format!("missing orbit direction: {clause}")); };
        let min_degrees = number_after(rest, "for ").ok_or_else(|| format!("missing orbit angle: {clause}"))?;
        let radius = number_after(rest, "at ").ok_or_else(|| format!("missing orbit radius: {clause}"))?;
        return Ok(IntentSpec::Orbit { landmark, direction, min_degrees, radius_m: radius });
    }
    if c.starts_with("hold ") && c.contains(" from the ") {
        let distance = number_after(c, "hold ").ok_or_else(|| format!("missing hold distance: {clause}"))?;
        let rest = c.split(" from the ").nth(1).ok_or_else(|| format!("missing landmark: {clause}"))?;
        let landmark = parse_landmark_prefix(rest)?;
        let duration = number_before(c, "second").ok_or_else(|| format!("missing hold duration: {clause}"))?;
        return Ok(IntentSpec::HoldAt { landmark, distance_m: distance, tolerance_m: 0.9, duration_s: duration });
    }
    if c.starts_with("return to the start") || c == "return home" {
        return Ok(IntentSpec::Return { tolerance_m: 0.8 });
    }
    Err(format!("unknown instruction clause: {clause}"))
}

fn parse_landmark_prefix(text: &str) -> Result<LandmarkDesc, String> {
    let re = Regex::new(r"^(red|blue|yellow|green) (circle|square|triangle|cross)").unwrap();
    let caps = re.captures(text).ok_or_else(|| format!("missing landmark in: {text}"))?;
    Ok(LandmarkDesc::new(
        Color::from_name(caps.get(1).unwrap().as_str()).unwrap(),
        Shape::from_name(caps.get(2).unwrap().as_str()).unwrap(),
    ))
}

fn number_before(text: &str, unit: &str) -> Option<f64> {
    let re = Regex::new(&format!(r"(\d+(?:\.\d+)?)\s*{}", unit)).ok()?;
    re.captures(text).and_then(|c| c.get(1)?.as_str().parse().ok())
}

fn number_after(text: &str, marker: &str) -> Option<f64> {
    let idx = text.find(marker)?;
    let rest = &text[idx + marker.len()..];
    let re = Regex::new(r"(\d+(?:\.\d+)?)").unwrap();
    re.captures(rest).and_then(|c| c.get(1)?.as_str().parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_each_family() {
        let cases = [
            "Move forward 3.0 metres.",
            "Turn left 90 degrees.",
            "Climb 2.0 metres.",
            "Hold position for 3.0 seconds.",
            "Approach the red circle to within 4.0 metres.",
            "Pass the blue square on the left.",
            "Orbit the green cross clockwise for 180 degrees at 3.5 metres.",
            "Hold 5.0 metres from the yellow triangle for 2.0 seconds.",
            "Return to the start.",
        ];
        for c in cases {
            let parsed = InstructionSpec::parse(c).unwrap_or_else(|e| panic!("{c}: {e}"));
            assert_eq!(parsed.intents.len(), 1);
        }
    }

    #[test]
    fn render_round_trips_composition() {
        let text = "Approach the red circle to within 4.0 metres, then pass the red circle on the left, then return to the start.";
        let parsed = InstructionSpec::parse(text).unwrap();
        assert_eq!(parsed.intents.len(), 3);
        let rendered = InstructionSpec::render(&parsed.intents);
        let reparsed = InstructionSpec::parse(&rendered).unwrap();
        assert_eq!(reparsed.intents, parsed.intents);
    }
}
