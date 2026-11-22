//! Evaluation results, paired comparisons, and reproducible reports.
//!
//! Reads the JSON files written by `scripts/eval_uavflow_test_5090.py` and
//! implements the metrics used in the thesis: success, NDTW, paired wins/losses,
//! McNemar exact test, paired bootstrap CI, and horizon bins.

use anyhow::{Context, Result};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalResult {
    pub task: String,
    pub success: bool,
    pub ndtw: f64,
    pub final_target_m: f64,
    pub steps: u64,
    pub path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalSummary {
    pub tasks: usize,
    pub successes: usize,
    pub success_rate: f64,
    pub success_ci: [f64; 2],
    pub mean_ndtw: f64,
    pub mean_final_target_m: f64,
    pub ndtw_ci: [f64; 2],
    pub by_horizon: BTreeMap<String, HorizonStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizonStats {
    pub tasks: usize,
    pub successes: usize,
    pub success_rate: f64,
    pub mean_ndtw: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedComparison {
    pub tasks: usize,
    pub baseline: EvalSummary,
    pub candidate: EvalSummary,
    pub success_delta_pp: f64,
    pub candidate_wins: usize,
    pub baseline_wins: usize,
    pub ties: usize,
    pub mcnemar_p: f64,
    pub ndtw_delta: f64,
    pub ndtw_t: f64,
    pub ndtw_p: f64,
    pub ndtw_delta_ci: [f64; 2],
}

fn as_f64(v: &serde_json::Value, key: &str) -> Option<f64> {
    v.get(key).and_then(|x| x.as_f64()).or_else(|| {
        v.get(key)
            .and_then(|x| x.as_bool())
            .map(|b| if b { 1.0 } else { 0.0 })
    })
}

/// Parse one result JSON, tolerating the two historical schemas.
pub fn parse_result(path: &Path) -> Result<EvalResult> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let v: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("json {}", path.display()))?;
    let task = v
        .get("task")
        .and_then(|x| x.as_str())
        .unwrap_or_else(|| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
        })
        .to_string();
    let r0 = v
        .get("rollouts")
        .and_then(|x| x.as_array())
        .and_then(|a| a.first());
    let success = as_f64(&v, "success")
        .or_else(|| r0.and_then(|r| as_f64(r, "success")))
        .unwrap_or(0.0)
        > 0.5;
    let ndtw = as_f64(&v, "ndtW")
        .or_else(|| r0.and_then(|r| as_f64(r, "ndtW")))
        .unwrap_or(0.0);
    let final_target_m = as_f64(&v, "final_target_m")
        .or_else(|| r0.and_then(|r| as_f64(r, "final_target_m")))
        .unwrap_or(f64::NAN);
    let steps = r0
        .and_then(|r| r.get("steps"))
        .and_then(|x| x.as_u64())
        .unwrap_or(0);
    Ok(EvalResult {
        task,
        success,
        ndtw,
        final_target_m,
        steps,
        path: Some(path.to_path_buf()),
    })
}

pub fn load_dir(dir: &Path) -> Result<Vec<EvalResult>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("read dir {}", dir.display()))? {
        let p = entry?.path();
        if p.extension().and_then(|s| s.to_str()) == Some("json")
            && p.file_name().and_then(|s| s.to_str()) != Some("summary.json")
        {
            out.push(parse_result(&p)?);
        }
    }
    out.sort_by(|a, b| a.task.cmp(&b.task));
    if out.is_empty() {
        anyhow::bail!("no result JSON files in {}", dir.display());
    }
    Ok(out)
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn wilson(k: usize, n: usize) -> [f64; 2] {
    if n == 0 {
        return [0.0, 1.0];
    }
    let p = k as f64 / n as f64;
    let z = 1.96;
    let d = 1.0 + z * z / n as f64;
    let c = (p + z * z / (2.0 * n as f64)) / d;
    let h = z * ((p * (1.0 - p) / n as f64) + z * z / (4.0 * n as f64 * n as f64)).sqrt() / d;
    [(c - h).max(0.0), (c + h).min(1.0)]
}

fn bootstrap_ci(v: &[f64], reps: usize, seed: u64) -> [f64; 2] {
    if v.is_empty() {
        return [0.0, 0.0];
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut means = Vec::with_capacity(reps);
    for _ in 0..reps {
        let mut s = 0.0;
        for _ in 0..v.len() {
            s += v[rng.gen_range(0..v.len())];
        }
        means.push(s / v.len() as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = means[((0.025 * reps as f64) as usize).min(reps - 1)];
    let hi = means[((0.975 * reps as f64) as usize).min(reps - 1)];
    [lo, hi]
}

fn horizon_bin(steps: u64) -> &'static str {
    if steps == 0 {
        "unknown"
    } else if steps <= 60 {
        "<=60"
    } else if steps <= 150 {
        "61-150"
    } else {
        ">150"
    }
}

pub fn summarize(results: &[EvalResult]) -> EvalSummary {
    let n = results.len();
    let k = results.iter().filter(|r| r.success).count();
    let nd = results.iter().map(|r| r.ndtw).collect::<Vec<_>>();
    let fin = results
        .iter()
        .map(|r| r.final_target_m)
        .filter(|x| x.is_finite())
        .collect::<Vec<_>>();
    let mut bins: BTreeMap<String, Vec<&EvalResult>> = BTreeMap::new();
    for r in results {
        bins.entry(horizon_bin(r.steps).to_string())
            .or_default()
            .push(r);
    }
    let by_horizon = bins
        .into_iter()
        .map(|(k, v)| {
            let kk = v.iter().filter(|r| r.success).count();
            (
                k,
                HorizonStats {
                    tasks: v.len(),
                    successes: kk,
                    success_rate: if v.is_empty() {
                        0.0
                    } else {
                        kk as f64 / v.len() as f64
                    },
                    mean_ndtw: mean(&v.iter().map(|r| r.ndtw).collect::<Vec<_>>()),
                },
            )
        })
        .collect();
    EvalSummary {
        tasks: n,
        successes: k,
        success_rate: if n == 0 { 0.0 } else { k as f64 / n as f64 },
        success_ci: wilson(k, n),
        mean_ndtw: mean(&nd),
        mean_final_target_m: mean(&fin),
        ndtw_ci: bootstrap_ci(&nd, 2000, 42),
        by_horizon,
    }
}

fn erf(x: f64) -> f64 {
    // Abramowitz-Stegun 7.1.26, accurate enough for a two-sided p-value.
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    if x >= 0.0 {
        y
    } else {
        -y
    }
}

fn normal_two_sided_p(t: f64) -> f64 {
    (2.0 * (1.0 - 0.5 * (1.0 + erf(t.abs() / std::f64::consts::SQRT_2)))).clamp(0.0, 1.0)
}

fn choose(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    (1..=k).fold(1.0, |acc, i| acc * (n - k + i) as f64 / i as f64)
}

/// McNemar exact two-sided p-value for paired binary outcomes.
pub fn mcnemar_exact(b: usize, c: usize) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    let m = b.min(c);
    let tail: f64 = (0..=m).map(|i| choose(n, i)).sum::<f64>() / 2f64.powi(n as i32);
    (2.0 * tail).min(1.0)
}

/// Compare two result directories on their common tasks.
pub fn compare_dirs(baseline: &Path, candidate: &Path) -> Result<PairedComparison> {
    let b = load_dir(baseline)?;
    let c = load_dir(candidate)?;
    compare_results(&b, &c)
}

pub fn compare_results(b: &[EvalResult], c: &[EvalResult]) -> Result<PairedComparison> {
    let bm: BTreeMap<&str, &EvalResult> = b.iter().map(|r| (r.task.as_str(), r)).collect();
    let cm: BTreeMap<&str, &EvalResult> = c.iter().map(|r| (r.task.as_str(), r)).collect();
    let common: Vec<&str> = bm.keys().filter(|k| cm.contains_key(*k)).copied().collect();
    if common.is_empty() {
        anyhow::bail!("no common tasks");
    }
    let bb: Vec<&EvalResult> = common.iter().map(|k| bm[k]).collect();
    let cc: Vec<&EvalResult> = common.iter().map(|k| cm[k]).collect();
    let bw = common
        .iter()
        .filter(|k| bm[*k].success && !cm[*k].success)
        .count();
    let cw = common
        .iter()
        .filter(|k| !bm[*k].success && cm[*k].success)
        .count();
    let ties = common.len() - bw - cw;
    let diffs: Vec<f64> = common.iter().map(|k| cm[*k].ndtw - bm[*k].ndtw).collect();
    let md = mean(&diffs);
    let sd = if diffs.len() > 1 {
        let var = diffs.iter().map(|d| (d - md).powi(2)).sum::<f64>() / (diffs.len() - 1) as f64;
        var.sqrt()
    } else {
        0.0
    };
    let se = if sd == 0.0 {
        0.0
    } else {
        sd / (diffs.len() as f64).sqrt()
    };
    let t = if se == 0.0 { 0.0 } else { md / se };
    let p = if se == 0.0 {
        1.0
    } else {
        normal_two_sided_p(t)
    };
    let bs = summarize(&bb.iter().map(|r| (*r).clone()).collect::<Vec<_>>());
    let cs = summarize(&cc.iter().map(|r| (*r).clone()).collect::<Vec<_>>());
    let delta_pp = 100.0 * (cs.success_rate - bs.success_rate);
    let ci = if se == 0.0 {
        [md, md]
    } else {
        [md - 1.96 * se, md + 1.96 * se]
    };
    Ok(PairedComparison {
        tasks: common.len(),
        baseline: bs,
        candidate: cs,
        success_delta_pp: delta_pp,
        candidate_wins: cw,
        baseline_wins: bw,
        ties,
        mcnemar_p: mcnemar_exact(cw, bw),
        ndtw_delta: md,
        ndtw_t: t,
        ndtw_p: p,
        ndtw_delta_ci: ci,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub generated_by: String,
    pub results: EvalSummary,
    pub tasks: Vec<EvalResult>,
}

pub fn write_report(dir: &Path, out_json: &Path) -> Result<EvalReport> {
    let results = load_dir(dir)?;
    let report = EvalReport {
        generated_by: "dronevlctl eval".into(),
        results: summarize(&results),
        tasks: results,
    };
    std::fs::write(out_json, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tempdir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("dronevl-eval-{}-{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }
    fn write_res(dir: &Path, task: &str, success: bool, ndtw: f64, steps: u64) {
        std::fs::write(dir.join(format!("{task}.json")), format!(r#"{{"task":"{task}","rollouts":[{{"steps":{steps},"success":{success},"ndtW":{ndtw},"final_target_m":1.0}}],"success":{},"ndtW":{ndtw}}}"#, if success {1} else {0})).unwrap();
    }
    #[test]
    fn parses_and_summarizes() {
        let d = tempdir("summary");
        write_res(&d, "a", true, 0.9, 50);
        write_res(&d, "b", false, 0.2, 200);
        let r = load_dir(&d).unwrap();
        let s = summarize(&r);
        assert_eq!(s.tasks, 2);
        assert_eq!(s.successes, 1);
        assert_eq!(s.by_horizon["<=60"].successes, 1);
        assert_eq!(s.by_horizon[">150"].success_rate, 0.0);
    }
    #[test]
    fn paired_mcnemar_and_ndtw() {
        let a = tempdir("paired-a");
        let b = tempdir("paired-b");
        write_res(&a, "a", true, 0.8, 50);
        write_res(&a, "b", false, 0.2, 50);
        write_res(&b, "a", true, 0.9, 50);
        write_res(&b, "b", true, 0.3, 50);
        let c = compare_dirs(&a, &b).unwrap();
        assert_eq!(c.tasks, 2);
        assert_eq!(c.candidate_wins, 1);
        assert_eq!(c.baseline_wins, 0);
        assert!((c.ndtw_delta - 0.1).abs() < 1e-9);
        assert!(c.mcnemar_p <= 1.0);
    }
}
