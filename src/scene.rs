//! Visual marker arena and a small deterministic software RGB/depth renderer.
//!
//! The renderer is intentionally simple but is a real pinhole camera over a 3D
//! scene: policies receive rendered pixels, not ground-truth marker poses. An
//! independent perception stage turns the image back into colour/shape
//! detections with measurement noise. Ground-truth poses remain evaluator-only.

use crate::geom::{body_to_ned, Camera, Vec3};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Color {
    Red,
    Blue,
    Yellow,
    Green,
}

impl Color {
    pub const ALL: [Color; 4] = [Color::Red, Color::Blue, Color::Yellow, Color::Green];
    pub fn rgb(self) -> [u8; 3] {
        match self {
            Color::Red => [214, 48, 45],
            Color::Blue => [42, 86, 214],
            Color::Yellow => [235, 190, 28],
            Color::Green => [44, 176, 74],
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Color::Red => "red",
            Color::Blue => "blue",
            Color::Yellow => "yellow",
            Color::Green => "green",
        }
    }
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "red" => Some(Color::Red),
            "blue" => Some(Color::Blue),
            "yellow" => Some(Color::Yellow),
            "green" => Some(Color::Green),
            _ => None,
        }
    }
    pub fn nearest(rgb: [f64; 3]) -> (Color, f64) {
        let mut best = Color::Red;
        let mut best_d = f64::INFINITY;
        for c in Color::ALL {
            let p = c.rgb();
            let d = (rgb[0] - p[0] as f64).powi(2)
                + (rgb[1] - p[1] as f64).powi(2)
                + (rgb[2] - p[2] as f64).powi(2);
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        (best, best_d.sqrt())
    }
}

impl std::fmt::Display for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Shape {
    Circle,
    Square,
    Triangle,
    Cross,
}

impl Shape {
    pub const ALL: [Shape; 4] = [Shape::Circle, Shape::Square, Shape::Triangle, Shape::Cross];
    pub fn name(self) -> &'static str {
        match self {
            Shape::Circle => "circle",
            Shape::Square => "square",
            Shape::Triangle => "triangle",
            Shape::Cross => "cross",
        }
    }
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "circle" => Some(Shape::Circle),
            "square" => Some(Shape::Square),
            "triangle" => Some(Shape::Triangle),
            "cross" => Some(Shape::Cross),
            _ => None,
        }
    }
    fn contains(self, x: f64, y: f64) -> bool {
        // x,y in [-0.5, 0.5], y down.
        match self {
            Shape::Circle => x * x + y * y <= 0.42 * 0.42,
            Shape::Square => x.abs() <= 0.34 && y.abs() <= 0.34,
            Shape::Triangle => {
                let top = Vec3::new(0.0, -0.45, 0.0);
                let bl = Vec3::new(-0.45, 0.40, 0.0);
                let br = Vec3::new(0.45, 0.40, 0.0);
                let p = Vec3::new(x, y, 0.0);
                let s = |a: Vec3, b: Vec3| (p.x - b.x) * (a.y - b.y) - (a.x - b.x) * (p.y - b.y);
                let d1 = s(top, bl);
                let d2 = s(bl, br);
                let d3 = s(br, top);
                let neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
                let pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
                !(neg && pos)
            }
            Shape::Cross => {
                let bar = 0.16;
                let arm = 0.44;
                (x.abs() <= bar && y.abs() <= arm) || (y.abs() <= bar && x.abs() <= arm)
            }
        }
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A visual marker: a flat panel standing at x = pos.x, facing -x.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Marker {
    pub id: String,
    pub color: Color,
    pub shape: Shape,
    pub pos: Vec3,
    pub size: f64,
}

impl Marker {
    pub fn inspection_pose(&self, standoff: f64) -> (Vec3, f64) {
        (Vec3::new(self.pos.x - standoff, self.pos.y, self.pos.z), 0.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub name: String,
    pub markers: Vec<Marker>,
    pub ground_z: f64,
}

pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
    pub depth: Vec<f32>,
}

impl Frame {
    pub fn pixel(&self, x: usize, y: usize) -> [u8; 3] {
        let i = (y * self.width + x) * 3;
        [self.rgb[i], self.rgb[i + 1], self.rgb[i + 2]]
    }
}

impl Scene {
    pub fn render(&self, camera: &Camera, position: Vec3, yaw: f64) -> Frame {
        let mut rgb = vec![0u8; camera.width * camera.height * 3];
        let mut depth = vec![f32::INFINITY; camera.width * camera.height];
        for v in 0..camera.height {
            for u in 0..camera.width {
                let d_body = camera.ray_body(u as f64 + 0.5, v as f64 + 0.5);
                let d = body_to_ned(d_body, yaw);
                let (color, t) = self.cast(position, d, camera);
                let i = v * camera.width + u;
                depth[i] = t as f32;
                let p = i * 3;
                rgb[p] = color[0];
                rgb[p + 1] = color[1];
                rgb[p + 2] = color[2];
            }
        }
        Frame {
            width: camera.width,
            height: camera.height,
            rgb,
            depth,
        }
    }

    fn cast(&self, o: Vec3, d: Vec3, _camera: &Camera) -> ([u8; 3], f64) {
        let mut best_t = f64::INFINITY;
        let mut best = self.sky(d);
        // Marker panels.
        for m in &self.markers {
            if d.x <= 1e-9 || o.x >= m.pos.x {
                continue;
            }
            let t = (m.pos.x - o.x) / d.x;
            if !(1e-6..best_t).contains(&t) {
                continue;
            }
            let hit = o.add(d.scale(t));
            let half = m.size * 0.5;
            if (hit.y - m.pos.y).abs() > half || (hit.z - m.pos.z).abs() > half {
                continue;
            }
            let lx = (hit.y - m.pos.y) / m.size;
            let ly = (hit.z - m.pos.z) / m.size;
            let c = self.panel_pixel(m, lx, ly);
            best_t = t;
            best = c;
        }
        // Ground plane z = ground_z.
        if d.z > 1e-9 {
            let t = (self.ground_z - o.z) / d.z;
            if t > 1e-6 && t < best_t {
                let hit = o.add(d.scale(t));
                best_t = t;
                best = self.ground(hit);
            }
        }
        (best, best_t)
    }

    fn panel_pixel(&self, m: &Marker, lx: f64, ly: f64) -> [u8; 3] {
        let border = lx.abs() > 0.47 || ly.abs() > 0.47;
        if border {
            return [72, 74, 80];
        }
        if m.shape.contains(lx, ly) {
            m.color.rgb()
        } else {
            [236, 235, 228]
        }
    }

    fn ground(&self, p: Vec3) -> [u8; 3] {
        let gx = p.x.rem_euclid(2.0);
        let gy = p.y.rem_euclid(2.0);
        let line = gx < 0.06 || gy < 0.06;
        let shade = if line { 96 } else { 112 };
        let far = (p.horizontal_norm() / 40.0).clamp(0.0, 0.35);
        let mix = |c: u8| ((c as f64) * (1.0 - far) + 205.0 * far) as u8;
        [mix(shade), mix(shade + 14), mix(shade - 10)]
    }

    fn sky(&self, d: Vec3) -> [u8; 3] {
        let up = (-d.z).clamp(0.0, 1.0);
        let t = up.powf(0.55);
        [
            (196.0 - 96.0 * t) as u8,
            (216.0 - 78.0 * t) as u8,
            (232.0 - 32.0 * t) as u8,
        ]
    }
}

/// A noisy colour/shape detection produced by the perception stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub color: Color,
    pub shape: Shape,
    pub bearing_rad: f64,
    pub elevation_rad: f64,
    pub range_m: f64,
    pub world_position: Vec3,
    pub pixel_area: usize,
    pub bbox_w: usize,
    pub bbox_h: usize,
    pub fill_ratio: f64,
    pub confidence: f64,
}

pub struct PerceptionConfig {
    pub min_area_px: usize,
    pub range_sigma_m: f64,
    pub bearing_sigma_rad: f64,
    pub dropout_p: f64,
    pub label_noise_p: f64,
}

impl Default for PerceptionConfig {
    fn default() -> Self {
        Self {
            min_area_px: 5,
            range_sigma_m: 0.06,
            bearing_sigma_rad: 0.006,
            dropout_p: 0.01,
            label_noise_p: 0.01,
        }
    }
}

/// Segment saturated marker pixels and fit one detection per connected
/// component. The returned world positions are estimates from the same noisy
/// bearing/range measurements available to the policy.
pub fn perceive<R: Rng>(
    frame: &Frame,
    camera: &Camera,
    position: Vec3,
    yaw: f64,
    cfg: &PerceptionConfig,
    rng: &mut R,
) -> Vec<Detection> {
    let w = frame.width;
    let h = frame.height;
    let mut mask = vec![false; w * h];
    for i in 0..w * h {
        let p = i * 3;
        let r = frame.rgb[p] as f64;
        let g = frame.rgb[p + 1] as f64;
        let b = frame.rgb[p + 2] as f64;
        let mx = r.max(g).max(b);
        let mn = r.min(g).min(b);
        if mx - mn > 55.0 && mx > 110.0 {
            mask[i] = true;
        }
    }
    let mut seen = vec![false; w * h];
    let mut detections = Vec::new();
    for start in 0..w * h {
        if !mask[start] || seen[start] {
            continue;
        }
        let mut q = VecDeque::new();
        q.push_back(start);
        seen[start] = true;
        let mut pixels = Vec::new();
        let (mut minx, mut maxx, mut miny, mut maxy) = (w, 0, h, 0);
        while let Some(idx) = q.pop_front() {
            pixels.push(idx);
            let x = idx % w;
            let y = idx / w;
            minx = minx.min(x);
            maxx = maxx.max(x);
            miny = miny.min(y);
            maxy = maxy.max(y);
            for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    continue;
                }
                let ni = ny as usize * w + nx as usize;
                if mask[ni] && !seen[ni] {
                    seen[ni] = true;
                    q.push_back(ni);
                }
            }
        }
        if pixels.len() < cfg.min_area_px {
            continue;
        }
        // Reject sky/ground regions that happen to be saturated: they are much
        // larger than a marker panel at the operating standoff.
        if pixels.len() > (w * h) / 3 {
            continue;
        }
        let bw = maxx - minx + 1;
        let bh = maxy - miny + 1;
        if bw < 2 || bh < 2 {
            continue;
        }
        // Partially visible markers are not reliable for shape or range.
        if minx == 0 || miny == 0 || maxx == w - 1 || maxy == h - 1 {
            continue;
        }
        let fill = pixels.len() as f64 / (bw * bh) as f64;
        if fill < 0.18 {
            continue;
        }
        let shape = if fill < 0.42 {
            Shape::Cross
        } else if fill < 0.63 {
            Shape::Triangle
        } else if fill < 0.90 {
            Shape::Circle
        } else {
            Shape::Square
        };
        let mut sum = [0.0f64; 3];
        for idx in &pixels {
            let p = idx * 3;
            sum[0] += frame.rgb[p] as f64;
            sum[1] += frame.rgb[p + 1] as f64;
            sum[2] += frame.rgb[p + 2] as f64;
        }
        let n = pixels.len() as f64;
        let mean = [sum[0] / n, sum[1] / n, sum[2] / n];
        let (mut color, color_dist) = Color::nearest(mean);
        if color_dist > 80.0 {
            continue;
        }
        if rng.gen::<f64>() < cfg.dropout_p {
            continue;
        }
        if rng.gen::<f64>() < cfg.label_noise_p {
            color = Color::ALL[rng.gen_range(0..4)];
        }
        let mut shape = shape;
        if rng.gen::<f64>() < cfg.label_noise_p {
            shape = Shape::ALL[rng.gen_range(0..4)];
        }
        let cx = (minx as f64 + maxx as f64) * 0.5 + 0.5;
        let cy = (miny as f64 + maxy as f64) * 0.5 + 0.5;
        let bearing = ((cx - w as f64 * 0.5) / camera.f).atan() + rng.gen::<f64>() * 0.0;
        let elevation = ((cy - h as f64 * 0.5) / camera.f).atan();
        // The coloured glyph occupies only part of the 1.5 m panel; use the
        // shape-specific extent so range is not biased by the white border.
        let extent = match shape {
            Shape::Circle => 0.84,
            Shape::Square => 0.68,
            Shape::Triangle => 0.90,
            Shape::Cross => 0.88,
        } * 1.5;
        let range = camera.f * extent / bw as f64;
        let bearing = bearing + gaussian(rng) * cfg.bearing_sigma_rad;
        let range = (range + gaussian(rng) * cfg.range_sigma_m).max(0.5);
        let d_body = Vec3::new(1.0, bearing.tan(), elevation.tan()).normalized();
        let world = position.add(body_to_ned(d_body.scale(range), yaw));
        let confidence = (pixels.len() as f64 / 120.0).clamp(0.15, 1.0);
        detections.push(Detection {
            color,
            shape,
            bearing_rad: bearing,
            elevation_rad: elevation,
            range_m: range,
            world_position: world,
            pixel_area: pixels.len(),
            bbox_w: bw,
            bbox_h: bh,
            fill_ratio: fill,
            confidence,
        });
    }
    // Left-to-right ordering is stable and useful for diagnostics.
    detections.sort_by(|a, b| a.bearing_rad.partial_cmp(&b.bearing_rad).unwrap());
    detections
}

fn gaussian<R: Rng>(rng: &mut R) -> f64 {
    // Box-Muller; deterministic per decision seed.
    let u1 = rng.gen::<f64>().max(1e-12);
    let u2 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn scene() -> Scene {
        Scene {
            name: "test".into(),
            ground_z: 0.0,
            markers: vec![Marker {
                id: "A".into(),
                color: Color::Red,
                shape: Shape::Circle,
                pos: Vec3::new(6.0, 0.0, -10.0),
                size: 1.5,
            }],
        }
    }

    #[test]
    fn renders_and_detects_a_marker() {
        let cam = Camera::new(96, 72, std::f64::consts::FRAC_PI_2);
        let frame = scene().render(&cam, Vec3::new(0.0, 0.0, -10.0), 0.0);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let d = perceive(
            &frame,
            &cam,
            Vec3::new(0.0, 0.0, -10.0),
            0.0,
            &PerceptionConfig::default(),
            &mut rng,
        );
        assert_eq!(d.len(), 1, "detections: {d:?}");
        assert_eq!(d[0].color, Color::Red);
        assert_eq!(d[0].shape, Shape::Circle);
        assert!((d[0].range_m - 6.0).abs() < 0.3, "range {}", d[0].range_m);
        assert!(d[0].bearing_rad.abs() < 0.02);
        assert!((d[0].world_position.x - 6.0).abs() < 0.4);
    }

    #[test]
    fn marker_outside_fov_is_not_detected() {
        let cam = Camera::new(96, 72, std::f64::consts::FRAC_PI_2);
        let frame = scene().render(&cam, Vec3::new(0.0, 0.0, -10.0), std::f64::consts::PI);
        let mut rng = ChaCha8Rng::seed_from_u64(8);
        let d = perceive(
            &frame,
            &cam,
            Vec3::new(0.0, 0.0, -10.0),
            std::f64::consts::PI,
            &PerceptionConfig::default(),
            &mut rng,
        );
        assert!(d.is_empty());
    }
}
