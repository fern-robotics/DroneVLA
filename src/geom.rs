//! NED geometry and angles. World frame is NED: x forward (north), y right
//! (east), z down. Yaw is clockwise-positive around +z, matching AirSim.

use serde::{Deserialize, Serialize};

pub const TAU: f64 = std::f64::consts::TAU;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
    pub fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
    pub fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
    pub fn scale(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
    pub fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }
    pub fn distance(self, o: Vec3) -> f64 {
        self.sub(o).norm()
    }
    pub fn horizontal_norm(self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
    pub fn normalized(self) -> Vec3 {
        let n = self.norm();
        if n < 1e-12 {
            Vec3::zero()
        } else {
            self.scale(1.0 / n)
        }
    }
    pub fn lerp(self, o: Vec3, t: f64) -> Vec3 {
        self.add(o.sub(self).scale(t))
    }
    pub fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
    pub fn from_array(a: [f64; 3]) -> Vec3 {
        Vec3::new(a[0], a[1], a[2])
    }
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl std::fmt::Display for Vec3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:.3}, {:.3}, {:.3})", self.x, self.y, self.z)
    }
}

/// Wrap an angle to (-pi, pi].
pub fn wrap_angle(a: f64) -> f64 {
    let mut v = (a + std::f64::consts::PI).rem_euclid(TAU) - std::f64::consts::PI;
    if v <= -std::f64::consts::PI {
        v += TAU;
    }
    v
}

/// Yaw-aligned body (forward, right, down) to NED.
pub fn body_to_ned(d: Vec3, yaw: f64) -> Vec3 {
    let (s, c) = yaw.sin_cos();
    Vec3::new(c * d.x - s * d.y, s * d.x + c * d.y, d.z)
}

/// NED to yaw-aligned body (forward, right, down).
pub fn ned_to_body(d: Vec3, yaw: f64) -> Vec3 {
    body_to_ned(d, -yaw)
}

pub fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

pub fn angle_between(a: Vec3, b: Vec3) -> f64 {
    let d = a.normalized().dot(b.normalized());
    d.clamp(-1.0, 1.0).acos()
}

/// Pinhole camera looking along body +x with image x right (+body y) and
/// image y down (+body z).
pub struct Camera {
    pub width: usize,
    pub height: usize,
    pub fov_h_rad: f64,
    pub f: f64,
}

impl Camera {
    pub fn new(width: usize, height: usize, fov_h_rad: f64) -> Self {
        let f = (width as f64 * 0.5) / (fov_h_rad * 0.5).tan();
        Self {
            width,
            height,
            fov_h_rad,
            f,
        }
    }

    /// Ray direction in body frame for continuous pixel coordinates.
    pub fn ray_body(&self, u: f64, v: f64) -> Vec3 {
        let cx = self.width as f64 * 0.5;
        let cy = self.height as f64 * 0.5;
        Vec3::new(1.0, (u - cx) / self.f, (v - cy) / self.f).normalized()
    }

    pub fn project_body(&self, p: Vec3) -> Option<(f64, f64, f64)> {
        if p.x <= 1e-6 {
            return None;
        }
        let u = self.width as f64 * 0.5 + self.f * p.y / p.x;
        let v = self.height as f64 * 0.5 + self.f * p.z / p.x;
        Some((u, v, p.x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_frame_rotates_with_yaw() {
        let f = body_to_ned(Vec3::new(1.0, 0.0, 0.0), std::f64::consts::FRAC_PI_2);
        assert!((f.x - 0.0).abs() < 1e-9);
        assert!((f.y - 1.0).abs() < 1e-9);
        let r = body_to_ned(Vec3::new(0.0, 1.0, 0.0), 0.0);
        assert!((r.y - 1.0).abs() < 1e-9);
        let back = ned_to_body(f, std::f64::consts::FRAC_PI_2);
        assert!((back.x - 1.0).abs() < 1e-9);
    }

    #[test]
    fn wrap_angle_is_canonical() {
        for k in -4..5 {
            let a = std::f64::consts::PI + k as f64 * TAU;
            let w = wrap_angle(a);
            assert!(w > -std::f64::consts::PI - 1e-12);
            assert!(w <= std::f64::consts::PI + 1e-12);
        }
        assert!((wrap_angle(3.0 * std::f64::consts::PI) - std::f64::consts::PI).abs() < 1e-9);
    }

    #[test]
    fn projection_round_trip() {
        let cam = Camera::new(128, 96, std::f64::consts::FRAC_PI_2);
        let p = Vec3::new(4.0, 1.0, 0.5);
        let (u, v, _) = cam.project_body(p).unwrap();
        let d = cam.ray_body(u, v);
        assert!(angle_between(d, p.normalized()) < 1e-9);
    }
}
