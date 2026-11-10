//! Aggregation, paired bootstrap uncertainty, figures, and the experiment report.

use crate::experiment::ExperimentConfig;
use crate::runner::EpisodeSummary;
use crate::train::GrpoLog;
use crate::uavflow::UavStats;
use anyhow::Result;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionStats {
    pub arm: String,
    pub condition: String,
    pub split: String,
    pub n: usize,
    pub success_rate: f64,
    pub success_ci: [f64; 2],
    pub intent_score: f64,
    pub intent_ci: [f64; 2],
    pub collision_rate: f64,
    pub intervention_rate: f64,
    pub invalid_rate: f64,
    pub efficiency: Option<f64>,
    pub mean_decisions: f64,
    pub mean_sim_time_s: f64,
    pub inference_p95_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedDiff {
    pub arm: String,
    pub comparison: String,
    pub split: String,
    pub metric: String,
    pub mean_diff: f64,
    pub ci: [f64; 2],
    pub n_pairs: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningPoint {
    pub arm: String,
    pub group: usize,
    pub reward: f64,
    pub success_rate: f64,
    pub intent_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Analysis {
    pub conditions: Vec<ConditionStats>,
    pub diffs: Vec<PairedDiff>,
    pub learning: Vec<LearningPoint>,
    pub n_results: usize,
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
}

fn wilson(k: usize, n: usize) -> [f64; 2] {
    if n == 0 { return [0.0, 1.0]; }
    let p = k as f64 / n as f64;
    let z = 1.96;
    let denom = 1.0 + z * z / n as f64;
    let center = (p + z * z / (2.0 * n as f64)) / denom;
    let half = z * ((p * (1.0 - p) / n as f64) + z * z / (4.0 * n as f64 * n as f64)).sqrt() / denom;
    [(center - half).max(0.0), (center + half).min(1.0)]
}

fn subset<'a>(rows: &'a [EpisodeSummary], arm: &str, condition: &str, ood: Option<bool>) -> Vec<&'a EpisodeSummary> {
    rows.iter()
        .filter(|r| r.arm == arm && r.condition == condition && (ood.is_none() || Some(r.ood) == ood))
        .collect()
}

pub fn analyze(results: &[EpisodeSummary], logs: &[(String, u64, GrpoLog)], _out: &Path) -> Result<Analysis> {
    let mut conditions = Vec::new();
    let arms = ["openvla", "pi0fast"];
    let conds = ["M0", "M1", "SFT+", "M2"];
    for arm in arms {
        for cond in conds {
            for (label, ood) in [("all", None), ("id", Some(false)), ("ood", Some(true))] {
                let rows = subset(results, arm, cond, ood);
                if rows.is_empty() { continue; }
                let n = rows.len();
                let successes = rows.iter().filter(|r| r.success).count();
                let intent: Vec<f64> = rows.iter().map(|r| r.intent_score).collect();
                let eff: Vec<f64> = rows.iter().filter_map(|r| r.successful_path_efficiency).collect();
                let ci = bootstrap_ci(&intent, 7);
                conditions.push(ConditionStats {
                    arm: arm.into(),
                    condition: cond.into(),
                    split: label.into(),
                    n,
                    success_rate: successes as f64 / n as f64,
                    success_ci: wilson(successes, n),
                    intent_score: mean(&intent),
                    intent_ci: ci,
                    collision_rate: rows.iter().filter(|r| r.had_collision).count() as f64 / n as f64,
                    intervention_rate: mean(&rows.iter().map(|r| r.safety_intervention_rate).collect::<Vec<_>>()),
                    invalid_rate: mean(&rows.iter().map(|r| r.invalid_action_rate).collect::<Vec<_>>()),
                    efficiency: if eff.is_empty() { None } else { Some(mean(&eff)) },
                    mean_decisions: mean(&rows.iter().map(|r| r.decisions as f64).collect::<Vec<_>>()),
                    mean_sim_time_s: mean(&rows.iter().map(|r| r.mission_sim_time_s).collect::<Vec<_>>()),
                    inference_p95_ms: mean(&rows.iter().map(|r| r.inference_ms_p95).collect::<Vec<_>>()),
                });
            }
        }
    }

    let mut diffs = Vec::new();
    for arm in arms {
        for (a, b, label) in [("M1", "M0", "M1-M0"), ("M2", "M1", "M2-M1"), ("M2", "SFT+", "M2-SFT+")] {
            for (split, ood) in [("all", None), ("id", Some(false)), ("ood", Some(true))] {
                for metric in ["success", "intent"] {
                    let (diff, ci, n) = paired_bootstrap(results, arm, a, b, ood, metric);
                    if n > 1 {
                        diffs.push(PairedDiff {
                            arm: arm.into(),
                            comparison: label.into(),
                            split: split.into(),
                            metric: metric.into(),
                            mean_diff: diff,
                            ci,
                            n_pairs: n,
                        });
                    }
                }
            }
        }
    }

    // Learning curves averaged over seeds per arm and group index.
    let mut by_key: BTreeMap<(String, usize), Vec<&crate::train::GroupLog>> = BTreeMap::new();
    for (name, _, log) in logs {
        let arm = name.split('-').next().unwrap_or("unknown").to_string();
        for g in &log.groups {
            by_key.entry((arm.clone(), g.group)).or_default().push(g);
        }
    }
    let learning = by_key
        .into_iter()
        .map(|((arm, group), gs)| LearningPoint {
            arm,
            group,
            reward: mean(&gs.iter().map(|g| g.reward_mean).collect::<Vec<_>>()),
            success_rate: mean(&gs.iter().map(|g| g.success_rate).collect::<Vec<_>>()),
            intent_score: mean(&gs.iter().map(|g| g.intent_score).collect::<Vec<_>>()),
        })
        .collect();

    Ok(Analysis {
        conditions,
        diffs,
        learning,
        n_results: results.len(),
    })
}

fn bootstrap_ci(values: &[f64], seed: u64) -> [f64; 2] {
    if values.len() < 2 { return [0.0, 0.0]; }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut means = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let mut s = 0.0;
        for _ in 0..values.len() {
            s += values[rng.gen_range(0..values.len())];
        }
        means.push(s / values.len() as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    [means[25], means[974]]
}

fn paired_bootstrap(
    results: &[EpisodeSummary],
    arm: &str,
    cond_a: &str,
    cond_b: &str,
    ood: Option<bool>,
    metric: &str,
) -> (f64, [f64; 2], usize) {
    let mut a: HashMap<(u64, String), f64> = HashMap::new();
    let mut b: HashMap<(u64, String), f64> = HashMap::new();
    for r in results.iter().filter(|r| r.arm == arm && (ood.is_none() || Some(r.ood) == ood)) {
        let key = (r.train_seed, r.episode_id.clone());
        let v = if metric == "success" { if r.success { 1.0 } else { 0.0 } } else { r.intent_score };
        if r.condition == cond_a { a.insert(key.clone(), v); }
        if r.condition == cond_b { b.insert(key, v); }
    }
    let mut by_episode: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for (key, av) in &a {
        if let Some(bv) = b.get(key) {
            by_episode.entry(key.1.clone()).or_default().push(av - bv);
        }
    }
    let episodes: Vec<(String, Vec<f64>)> = by_episode.into_iter().collect();
    if episodes.len() < 2 { return (0.0, [0.0, 0.0], episodes.len()); }
    let point = mean(&episodes.iter().flat_map(|(_, v)| v.clone()).collect::<Vec<_>>());
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let mut samples = Vec::with_capacity(2000);
    for _ in 0..2000 {
        let mut vals = Vec::new();
        for _ in 0..episodes.len() {
            let (_, v) = &episodes[rng.gen_range(0..episodes.len())];
            vals.extend(v);
        }
        samples.push(mean(&vals));
    }
    samples.sort_by(|x, y| x.partial_cmp(y).unwrap());
    (point, [samples[50], samples[1949]], episodes.len())
}

pub fn write_report(out: &Path, a: &Analysis, stats: &UavStats, cfg: &ExperimentConfig) -> Result<()> {
    std::fs::write(out.join("metrics.json"), serde_json::to_string_pretty(a)?)?;
    let mut md = String::new();
    md.push_str("# DroneVL intent-adaptation experiment\n\n");
    md.push_str("## Protocol\n\n");
    md.push_str(&format!(
        "- Conditions: M0, M1 (LoRA-SFT analogue), SFT+ (continued SFT), M2 (GRPO from M1)\n- Arms: OpenVLA 256-bin delta4 tokens; pi0-FAST DCT action-chunk tokens\n- Seeds per arm: {}\n- SFT demonstrations: actual UAV-Flow-Sim, {} episodes / {} frames (q01 {:?}, q99 {:?})\n- GRPO: {} groups x {} rollouts on the AirSimNH training split\n- Evaluation: frozen AirSimNH test split, paired episodes; ID and held-out intent combinations reported separately\n\n",
        cfg.seeds.len(), stats.episodes_selected, stats.frames, stats.q01, stats.q99, cfg.groups, cfg.rollouts
    ));
    md.push_str("The learned component is a goal-conditioned linear motion policy standing in for the VLA; perception, intent planning, safety gate, and intent predicates are fixed across conditions. Results are a surrogate-pilot measurement of the protocol, not OpenVLA/pi0-FAST benchmark numbers.\n\n");
    md.push_str("## Success and intent adherence\n\n");
    md.push_str("| Arm | Condition | Split | N | Success | 95% CI | Intent score | Collision | Safety interv. |\n|---|---|---|---|---|---|---|---|---|\n");
    for c in &a.conditions {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {:.1}% | [{:.1}, {:.1}] | {:.3} | {:.1}% | {:.3} |\n",
            c.arm, c.condition, c.split, c.n, 100.0 * c.success_rate, 100.0 * c.success_ci[0], 100.0 * c.success_ci[1], c.intent_score, 100.0 * c.collision_rate, c.intervention_rate
        ));
    }
    md.push_str("\n## Paired differences (episode-paired bootstrap, 95% CI)\n\n");
    md.push_str("| Arm | Comparison | Split | Metric | Δ | 95% CI | Pairs |\n|---|---|---|---|---|---|---|\n");
    for d in &a.diffs {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {:+.3} | [{:+.3}, {:+.3}] | {} |\n",
            d.arm, d.comparison, d.split, d.metric, d.mean_diff, d.ci[0], d.ci[1], d.n_pairs
        ));
    }
    md.push_str("\n![success](figures/success.svg)\n\n![intent](figures/intent.svg)\n\n");
    md.push_str("## GRPO learning curve\n\n![learning](figures/learning.svg)\n\n");
    md.push_str("## Interpretation limits\n\n- The test split was frozen before training, but the arena and thresholds are development choices.\n- The surrogate policy consumes structured goal/perception features rather than raw VLM embeddings; cross-model tokenization differences are represented by the two action codecs.\n- Aviation safety claims are out of scope; collisions are ground/geofence contacts in a kinematic simulator.\n");
    std::fs::write(out.join("report.md"), md)?;
    write_svgs(out, a)?;
    Ok(())
}

fn write_svgs(out: &Path, a: &Analysis) -> Result<()> {
    std::fs::create_dir_all(out.join("figures"))?;
    let arms = ["openvla", "pi0fast"];
    let conds = ["M0", "M1", "SFT+", "M2"];
    // Success chart (all split).
    let mut bars = Vec::new();
    for arm in arms {
        for cond in conds {
            if let Some(c) = a.conditions.iter().find(|c| c.arm == arm && c.condition == cond && c.split == "all") {
                bars.push((format!("{arm}/{cond}"), c.success_rate, c.success_ci));
            }
        }
    }
    std::fs::write(out.join("figures/success.svg"), bar_svg("Test success rate", &bars))?;
    let mut intent_bars = Vec::new();
    for arm in arms {
        for cond in conds {
            if let Some(c) = a.conditions.iter().find(|c| c.arm == arm && c.condition == cond && c.split == "all") {
                intent_bars.push((format!("{arm}/{cond}"), c.intent_score, c.intent_ci));
            }
        }
    }
    std::fs::write(out.join("figures/intent.svg"), bar_svg("Intent adherence score", &intent_bars))?;
    std::fs::write(out.join("figures/learning.svg"), learning_svg(&a.learning))?;
    Ok(())
}

fn bar_svg(title: &str, bars: &[(String, f64, [f64; 2])]) -> String {
    let w = 760.0;
    let h = 380.0;
    let left = 60.0;
    let bottom = 320.0;
    let plot_h = 250.0;
    let bw = (w - left - 30.0) / bars.len() as f64 * 0.72;
    let mut s = format!(
        "<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}'><rect width='100%' height='100%' fill='white'/><text x='{left}' y='28' font-family='sans-serif' font-size='18'>{title}</text>"
    );
    for i in 0..=4 {
        let y = bottom - plot_h * i as f64 / 4.0;
        s.push_str(&format!("<line x1='{left}' y1='{y}' x2='{}' y2='{y}' stroke='#ddd'/><text x='{}' y='{}' font-family='sans-serif' font-size='11' text-anchor='end'>{:.1}</text>", w - 20.0, left - 6.0, y + 4.0, i as f64 / 4.0));
    }
    for (i, (label, value, ci)) in bars.iter().enumerate() {
        let x = left + 20.0 + i as f64 * (w - left - 40.0) / bars.len() as f64;
        let bh = plot_h * value.clamp(0.0, 1.0);
        s.push_str(&format!("<rect x='{x}' y='{}' width='{bw}' height='{bh}' fill='#3b6ea5'/>", bottom - bh));
        let y1 = bottom - plot_h * ci[0];
        let y2 = bottom - plot_h * ci[1];
        s.push_str(&format!("<line x1='{}' y1='{y1}' x2='{}' y2='{y1}' stroke='#222'/><line x1='{}' y1='{y2}' x2='{}' y2='{y2}' stroke='#222'/><line x1='{}' y1='{y1}' x2='{}' y2='{y2}' stroke='#222'/>", x + bw / 2.0, x + bw / 2.0 + 10.0, x + bw / 2.0, x + bw / 2.0 + 10.0, x + bw / 2.0 + 5.0, x + bw / 2.0 + 5.0));
        s.push_str(&format!("<text x='{}' y='{}' font-family='sans-serif' font-size='9' transform='rotate(-35 {} {})' text-anchor='end'>{}</text>", x + bw / 2.0, bottom + 8.0, x + bw / 2.0, bottom + 8.0, label));
    }
    s.push_str("</svg>");
    s
}

fn learning_svg(points: &[LearningPoint]) -> String {
    let w = 760.0;
    let h = 360.0;
    let left = 60.0;
    let bottom = 300.0;
    let plot_w = 650.0;
    let plot_h = 240.0;
    let max_g = points.iter().map(|p| p.group).max().unwrap_or(1).max(1);
    let mut s = format!("<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}'><rect width='100%' height='100%' fill='white'/><text x='{left}' y='28' font-family='sans-serif' font-size='18'>GRPO learning curve (group reward)</text>");
    for (arm, color) in [("openvla", "#3b6ea5"), ("pi0fast", "#c46a2b")] {
        let pts: Vec<&LearningPoint> = points.iter().filter(|p| p.arm == arm).collect();
        if pts.len() < 2 { continue; }
        let mut path = String::new();
        for (i, p) in pts.iter().enumerate() {
            let x = left + plot_w * p.group as f64 / max_g as f64;
            let y = bottom - plot_h * p.reward.clamp(-1.0, 2.0) / 2.0;
            path.push_str(&format!("{}{:.1},{:.1} ", if i == 0 { "M" } else { "L" }, x, y));
        }
        s.push_str(&format!("<path d='{path}' fill='none' stroke='{color}' stroke-width='2'/><text x='{}' y='{}' font-family='sans-serif' font-size='12' fill='{color}'>{arm}</text>", left + plot_w - 90.0, 45.0 + if arm == "openvla" { 0.0 } else { 16.0 }));
    }
    s.push_str("<line x1='60' y1='300' x2='710' y2='300' stroke='#333'/></svg>");
    s
}
