//! DroneVL-RS: a Rust instantiation of the revised DroneVL thesis experiment.
//!
//! The experiment compares M0, M1, continued SFT, and M2 (GRPO after M1) for
//! two action-tokenization interfaces (OpenVLA 256-bin and pi0-FAST DCT chunk).
//! SFT demonstrations are the actual UAV-Flow-Sim trajectories; GRPO rollouts
//! and evaluation use the AirSimNH-style visual landmark arena.
//!
//! The learned policy is a compact goal-conditioned linear controller that
//! stands in for the VLA. Perception, intent planning, safety gate, simulator,
//! and intent evaluator are fixed across conditions and live in Rust.

pub mod action;
pub mod analysis;
pub mod cache;
pub mod eval;
pub mod experiment;
pub mod experiment_plan;
pub mod geom;
pub mod instruction;
pub mod oracle;
pub mod orchestrator;
pub mod planner;
pub mod policy;
pub mod proxy;
pub mod reward;
pub mod runner;
pub mod safety;
pub mod scene;
pub mod sim;
pub mod task;
pub mod tracker;
pub mod train;
pub mod uavflow;
pub mod unrealcv;

pub const SCHEMA_VERSION: &str = "2.0";
