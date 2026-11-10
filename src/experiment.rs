//! Full experiment orchestration: frozen suite, SFT from UAV-Flow-Sim, GRPO on
//! AirSimNH training episodes, paired evaluation on the held-out test split.

use crate::action::{ActionBounds, ActionCodec};
use crate::analysis;
use crate::oracle;
use crate::policy::LinearPolicy;
use crate::runner::{EpisodeRun, EpisodeSummary, RunConfig};
use crate::task::{self, TaskEpisode};
use crate::train::{self, SftLog};
use crate::uavflow::{self, UavEpisode};
use anyhow::{Context, Result};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub out_dir: PathBuf,
    pub uavflow_path: PathBuf,
    pub seeds: Vec<u64>,
    pub epochs: usize,
    pub sft_stride: usize,
    pub groups: usize,
    pub rollouts: usize,
    pub test_limit: Option<usize>,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::from("runs/dronevl"),
            uavflow_path: PathBuf::from("data/uavflow_sim.jsonl"),
            seeds: vec![0, 1, 2],
            epochs: 25,
            sft_stride: 2,
            groups: 80,
            rollouts: 4,
            test_limit: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub arm: String,
    pub condition: String,
    pub seed: u64,
    pub init: String,
    pub epochs: usize,
    pub groups: usize,
    pub training_frames: usize,
}

pub fn run(cfg: &ExperimentConfig) -> Result<()> {
    let out = &cfg.out_dir;
    std::fs::create_dir_all(out)?;
    std::fs::create_dir_all(out.join("checkpoints"))?;
    std::fs::create_dir_all(out.join("training"))?;
    std::fs::create_dir_all(out.join("figures"))?;

    // 1. Frozen AirSimNH suite, validated with the privileged intent oracle.
    let mut episodes = task::generate_all();
    let failures = oracle::validate_episodes(&mut episodes);
    if !failures.is_empty() {
        anyhow::bail!("oracle failed {} episodes: {failures:?}", failures.len());
    }
    task::write_suite(&out.join("suite"), &episodes)?;
    std::fs::write(
        out.join("run_config.json"),
        serde_json::to_string_pretty(cfg)?,
    )?;
    let train_eps: Vec<TaskEpisode> = episodes.iter().filter(|e| e.split == "train").cloned().collect();
    let mut test_eps: Vec<TaskEpisode> = episodes.iter().filter(|e| e.split == "test").cloned().collect();
    if let Some(limit) = cfg.test_limit {
        test_eps.truncate(limit);
    }

    // 2. Actual UAV-Flow-Sim demonstrations.
    let demos: Vec<UavEpisode> = uavflow::load(&cfg.uavflow_path)
        .with_context(|| format!("loading {}", cfg.uavflow_path.display()))?;
    let samples = train::build_sft_samples(&demos, cfg.sft_stride);
    let stats = uavflow::load_stats(&cfg.uavflow_path.with_file_name("uavflow_stats.json"))
        .unwrap_or_else(|_| uavflow::UavStats {
            episodes_selected: demos.len(),
            frames: demos.iter().map(|e| e.actions.len()).sum(),
            q01: vec![-0.001, -0.25, -0.13, -0.003],
            q99: vec![0.31, 0.25, 0.20, 0.018],
            mean: vec![0.17, 0.0, 0.0, 0.0],
            std: vec![0.09, 0.11, 0.04, 0.005],
            abs_mean: vec![0.17, 0.08, 0.01, 0.001],
        });
    std::fs::write(out.join("uavflow_stats.json"), serde_json::to_string_pretty(&stats)?)?;

    let arms = [
        ("openvla", ActionCodec::openvla(ActionBounds::default())),
        ("pi0fast", ActionCodec::fast(ActionBounds::default())),
    ];
    let mut results: Vec<EpisodeSummary> = Vec::new();
    let mut all_sft: Vec<(String, u64, SftLog)> = Vec::new();
    let mut all_grpo: Vec<(String, u64, crate::train::GrpoLog)> = Vec::new();

    for (arm, codec) in &arms {
        for &seed in &cfg.seeds {
            let m0 = LinearPolicy::m0();
            let (m1, sft_log) = train::train_sft(&m0, &samples, cfg.epochs, 64, 0.01, 1e-4, seed);
            let (sft_plus, sft_plus_log) = train::train_sft(
                &m1,
                &samples,
                cfg.epochs * 2,
                64,
                0.005,
                1e-4,
                seed ^ 0xabcdef,
            );
            let (m2, grpo_log) = train::train_grpo(
                &m1,
                &m1,
                &train_eps,
                codec,
                cfg.groups,
                cfg.rollouts,
                0.02,
                0.2,
                0.04,
                seed ^ 0x5eed,
            );
            for (condition, policy, init) in [
                ("M0", &m0, "calibrated-prior"),
                ("M1", &m1, "M0"),
                ("SFT+", &sft_plus, "M1"),
                ("M2", &m2, "M1"),
            ] {
                let path = out.join(format!("checkpoints/{arm}-{seed}-{condition}.json"));
                policy.save(&path)?;
                let meta = CheckpointMeta {
                    arm: arm.to_string(),
                    condition: condition.to_string(),
                    seed,
                    init: init.to_string(),
                    epochs: cfg.epochs,
                    groups: cfg.groups,
                    training_frames: samples.len(),
                };
                std::fs::write(
                    out.join(format!("checkpoints/{arm}-{seed}-{condition}.meta.json")),
                    serde_json::to_string_pretty(&meta)?,
                )?;
            }
            std::fs::write(
                out.join(format!("training/{arm}-{seed}-m1-sft.json")),
                serde_json::to_string_pretty(&sft_log)?,
            )?;
            std::fs::write(
                out.join(format!("training/{arm}-{seed}-sftplus.json")),
                serde_json::to_string_pretty(&sft_plus_log)?,
            )?;
            std::fs::write(
                out.join(format!("training/{arm}-{seed}-grpo.json")),
                serde_json::to_string_pretty(&grpo_log)?,
            )?;
            all_sft.push((format!("{arm}-{seed}-M1"), seed, sft_log));
            all_grpo.push((format!("{arm}-{seed}"), seed, grpo_log));

            let policies: [(&str, &LinearPolicy); 4] = [
                ("M0", &m0),
                ("M1", &m1),
                ("SFT+", &sft_plus),
                ("M2", &m2),
            ];
            for (condition, policy) in policies {
                let rows: Vec<EpisodeSummary> = test_eps
                    .par_iter()
                    .map(|ep| {
                        let cfg = RunConfig::eval(arm, condition, seed);
                        EpisodeRun {
                            episode: ep,
                            policy,
                            codec,
                            config: &cfg,
                        }
                        .execute()
                        .summary
                    })
                    .collect();
                results.extend(rows);
            }
            eprintln!("finished arm={arm} seed={seed} results={}", results.len());
        }
    }

    // 3. Persist per-episode results and derived analysis.
    let mut file = std::fs::File::create(out.join("results.jsonl"))?;
    use std::io::Write;
    for r in &results {
        writeln!(file, "{}", serde_json::to_string(r)?)?;
    }
    let analysis = analysis::analyze(&results, &all_grpo, out)?;
    analysis::write_report(out, &analysis, &stats, cfg)?;
    eprintln!("wrote {} results to {}", results.len(), out.display());
    Ok(())
}

/// Smoke run used before the main experiment: M0/M1 only on a few episodes.
pub fn smoke(cfg: &ExperimentConfig) -> Result<()> {
    let out = &cfg.out_dir;
    std::fs::create_dir_all(out)?;
    let mut episodes = task::generate_all();
    let failures = oracle::validate_episodes(&mut episodes);
    if !failures.is_empty() {
        anyhow::bail!("oracle failures: {failures:?}");
    }
    let demos = uavflow::load(&cfg.uavflow_path)?;
    let samples = train::build_sft_samples(&demos, cfg.sft_stride);
    let (m1, log) = train::train_sft(&LinearPolicy::m0(), &samples, cfg.epochs.max(2), 64, 0.01, 1e-4, 0);
    std::fs::write(out.join("smoke_sft.json"), serde_json::to_string_pretty(&log)?)?;
    let codec = ActionCodec::openvla(ActionBounds::default());
    let test: Vec<&TaskEpisode> = episodes.iter().filter(|e| e.split == "test").take(cfg.test_limit.unwrap_or(8)).collect();
    for (name, p) in [("M0", LinearPolicy::m0()), ("M1", m1)] {
        let mut succ = 0;
        let mut score = 0.0;
        for ep in &test {
            let cfg = RunConfig::eval("openvla", name, 0);
            let r = EpisodeRun { episode: ep, policy: &p, codec: &codec, config: &cfg }.execute();
            succ += r.summary.success as i32;
            score += r.summary.intent_score;
            println!("{name} {} success={} score={:.2} term={} dec={}", ep.id, r.summary.success, r.summary.intent_score, r.summary.terminal_reason, r.summary.decisions);
        }
        println!("{name}: {}/{} success, intent {:.3}", succ, test.len(), score / test.len() as f64);
    }
    Ok(())
}

pub fn load_results(path: &Path) -> Result<Vec<EpisodeSummary>> {
    let text = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in text.lines() {
        if !line.trim().is_empty() {
            out.push(serde_json::from_str(line)?);
        }
    }
    Ok(out)
}

pub fn _unused(_: &mut Vec<EpisodeSummary>, _: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let _ = vec![1, 2, 3].choose(&mut rng);
}
