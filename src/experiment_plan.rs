//! Machine-readable plan for thesis experiments 1–8.
//!
//! This does not train anything. It turns the experiment table into runnable
//! specifications so the Rust orchestrator can schedule, resume, and report
//! ablations without a hand-maintained shell script per condition.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedRun {
    pub id: String,
    pub experiment: u8,
    pub question: String,
    pub factor: String,
    pub level: String,
    pub config: BTreeMap<String, String>,
    pub evaluation: String,
    pub runs: usize,
    pub cost_h: f64,
    pub status: String,
    pub priority: String,
    pub command: String,
}

fn cfg(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn train_cmd(bc: &str, anchor: &str, beta: &str, lr: &str) -> String {
    format!("python scripts/train_b0_eggrpo.py --groups {{groups}} --init-adapter {{adapter}} --out {{out}} --max-groups 150 --recompute-old --expert-bc-weight {bc} --expert-reward {anchor} --beta {beta} --lr {lr}")
}

fn online_cmd(rounds: &str, per: &str) -> String {
    format!("env TAG={{tag}} N_ROUNDS={rounds} PER={per} bash scripts/run_online_eggrpo.sh")
}

pub fn plan() -> Vec<PlannedRun> {
    let mut r = Vec::new();
    // 1. Main fixed-244 comparison.
    for (id, level, cmd) in [
        (
            "1-M0",
            "M0",
            "python scripts/eval_uavflow_test_5090.py --limit 244 --out {out}",
        ),
        (
            "1-OFF",
            "offline150",
            "python scripts/eval_uavflow_test_5090.py --limit 244 --adapter {adapter} --out {out}",
        ),
        (
            "1-ON",
            "online150",
            "python scripts/eval_uavflow_test_5090.py --limit 244 --adapter {adapter} --out {out}",
        ),
    ] {
        r.push(PlannedRun {
            id: id.into(),
            experiment: 1,
            question: "Does EG-GRPO beat M0?".into(),
            factor: "policy".into(),
            level: level.into(),
            config: cfg(&[("eval_tasks", "244")]),
            evaluation: "fixed244".into(),
            runs: 1,
            cost_h: 1.0,
            status: "done".into(),
            priority: "high".into(),
            command: cmd.into(),
        });
    }
    // 2. Paired online-vs-offline report.
    r.push(PlannedRun {
        id: "2-PAIRED".into(),
        experiment: 2,
        question: "Does online beat offline?".into(),
        factor: "policy".into(),
        level: "offline150 vs online150".into(),
        config: cfg(&[("paired", "true")]),
        evaluation: "fixed244".into(),
        runs: 1,
        cost_h: 0.0,
        status: "done".into(),
        priority: "high".into(),
        command: "dronevlctl eval-compare --baseline {offline_dir} --candidate {online_dir}".into(),
    });
    // 3. BC.
    for bc in ["0.0", "0.2", "0.5"] {
        r.push(PlannedRun {
            id: format!("3-BC-{bc}"),
            experiment: 3,
            question: "Does expert BC matter?".into(),
            factor: "expert_bc_weight".into(),
            level: bc.into(),
            config: cfg(&[
                ("expert_bc_weight", bc),
                ("expert_reward", "2.0"),
                ("beta", "0.0"),
                ("lr", "1e-5"),
            ]),
            evaluation: "screen50".into(),
            runs: 1,
            cost_h: 2.75,
            status: "pending".into(),
            priority: "high".into(),
            command: train_cmd(bc, "2.0", "0.0", "1e-5"),
        });
    }
    // 4. Expert anchor.
    for (name, val) in [("on", "2.0"), ("off", "0.0")] {
        r.push(PlannedRun {
            id: format!("4-ANCHOR-{name}"),
            experiment: 4,
            question: "Does expert reward anchor matter?".into(),
            factor: "expert_reward".into(),
            level: name.into(),
            config: cfg(&[
                ("expert_reward", val),
                ("expert_bc_weight", "0.2"),
                ("beta", "0.0"),
                ("lr", "1e-5"),
            ]),
            evaluation: "screen50".into(),
            runs: 1,
            cost_h: 2.75,
            status: "pending".into(),
            priority: "high".into(),
            command: train_cmd("0.2", val, "0.0", "1e-5"),
        });
    }
    // 5. KL.
    for beta in ["0.0", "0.01"] {
        r.push(PlannedRun {
            id: format!("5-KL-{beta}"),
            experiment: 5,
            question: "Does KL matter?".into(),
            factor: "beta".into(),
            level: beta.into(),
            config: cfg(&[
                ("beta", beta),
                ("expert_bc_weight", "0.2"),
                ("expert_reward", "2.0"),
                ("lr", "1e-5"),
            ]),
            evaluation: "screen50".into(),
            runs: 1,
            cost_h: 2.75,
            status: "pending".into(),
            priority: "medium".into(),
            command: train_cmd("0.2", "2.0", beta, "1e-5"),
        });
    }
    // 6. Learning rate.
    for lr in ["5e-6", "1e-5", "2e-5"] {
        r.push(PlannedRun {
            id: format!("6-LR-{lr}"),
            experiment: 6,
            question: "Does learning rate matter?".into(),
            factor: "lr".into(),
            level: lr.into(),
            config: cfg(&[
                ("lr", lr),
                ("expert_bc_weight", "0.2"),
                ("expert_reward", "2.0"),
                ("beta", "0.0"),
            ]),
            evaluation: "screen50".into(),
            runs: 1,
            cost_h: 2.75,
            status: "pending".into(),
            priority: "high".into(),
            command: train_cmd("0.2", "2.0", "0.0", lr),
        });
    }
    // 7. Data scale.
    for (name, n, cmd, cost) in [
        (
            "7-DS-OFF150",
            "150",
            train_cmd("0.2", "2.0", "0.0", "1e-5"),
            2.75,
        ),
        (
            "7-DS-OFF300",
            "300",
            train_cmd("0.2", "2.0", "0.0", "1e-5"),
            5.0,
        ),
        ("7-DS-ON150", "150", online_cmd("15", "10"), 6.75),
        ("7-DS-ON300", "300", online_cmd("15", "20"), 13.0),
    ] {
        r.push(PlannedRun {
            id: name.into(),
            experiment: 7,
            question: "Does data scale?".into(),
            factor: "data".into(),
            level: n.into(),
            config: cfg(&[("groups", n), ("eval_tasks", "50")]),
            evaluation: "screen50".into(),
            runs: 1,
            cost_h: cost,
            status: "pending".into(),
            priority: "high".into(),
            command: cmd,
        });
    }
    // 8. Online refresh schedule.
    for (name, rounds, per) in [
        ("5x30", "5", "30"),
        ("15x10", "15", "10"),
        ("30x5", "30", "5"),
    ] {
        let mut c = cfg(&[("rounds", rounds), ("per_round", per), ("updates", "150")]);
        c.insert("eval_tasks".into(), "50".into());
        r.push(PlannedRun {
            id: format!("8-ON-{name}"),
            experiment: 8,
            question: "Does online refresh frequency matter?".into(),
            factor: "schedule".into(),
            level: format!("{rounds}x{per}"),
            config: c,
            evaluation: "screen50".into(),
            runs: 1,
            cost_h: 6.75,
            status: "pending".into(),
            priority: "medium".into(),
            command: online_cmd(rounds, per),
        });
    }
    r
}

pub fn write_csv(path: &Path) -> Result<()> {
    let runs = plan();
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    use std::io::Write;
    writeln!(
        w,
        "id,experiment,question,factor,level,runs,cost_h,status,priority,evaluation,command"
    )?;
    for x in runs {
        writeln!(
            w,
            "{},{},{},{},{},{},{},{},{},{},{}",
            x.id,
            x.experiment,
            q(&x.question),
            q(&x.factor),
            q(&x.level),
            x.runs,
            x.cost_h,
            x.status,
            x.priority,
            x.evaluation,
            q(&x.command)
        )?;
    }
    Ok(())
}

fn q(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

pub fn write_json(path: &Path) -> Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(&plan())?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn covers_all_eight_experiments() {
        let p = plan();
        let ids: std::collections::BTreeSet<u8> = p.iter().map(|x| x.experiment).collect();
        assert_eq!(ids, (1u8..=8).collect());
        assert!(p.iter().any(|x| x.id == "3-BC-0.2"));
        assert!(p.iter().any(|x| x.id == "6-LR-2e-5"));
        assert!(p.iter().any(|x| x.id == "8-ON-30x5"));
        assert!(p.iter().all(|x| !x.command.is_empty()));
    }
}
