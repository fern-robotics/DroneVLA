use clap::{Parser, Subcommand};
use dronevl::experiment::{self, ExperimentConfig};
use dronevl::task;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "dronevl", about = "DroneVL intent-adaptation experiment (Rust)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Write the frozen AirSimNH intent suite and validate it with the oracle.
    Gen {
        #[arg(long, default_value = "runs/suite")]
        out: PathBuf,
    },
    /// Train the UAV-Flow-Sim SFT checkpoint (M1) and save it.
    M1 {
        #[arg(long, default_value = "data/uavflow_sim.jsonl")]
        uavflow: PathBuf,
        #[arg(long, default_value = "runs/m1.json")]
        out: PathBuf,
        #[arg(long, default_value_t = 25)]
        epochs: usize,
        #[arg(long, default_value_t = 2)]
        stride: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Quick M0/M1 check before the main run.
    Smoke {
        #[arg(long, default_value = "runs/smoke")]
        out: PathBuf,
        #[arg(long, default_value = "data/uavflow_sim.jsonl")]
        uavflow: PathBuf,
        #[arg(long, default_value_t = 4)]
        epochs: usize,
        #[arg(long, default_value_t = 6)]
        test_limit: usize,
    },
    /// Full M0/M1/SFT+/M2 experiment on both action interfaces.
    Experiment {
        #[arg(long, default_value = "runs/experiment")]
        out: PathBuf,
        #[arg(long, default_value = "data/uavflow_sim.jsonl")]
        uavflow: PathBuf,
        #[arg(long, default_value = "0,1,2")]
        seeds: String,
        #[arg(long, default_value_t = 25)]
        epochs: usize,
        #[arg(long, default_value_t = 2)]
        sft_stride: usize,
        #[arg(long, default_value_t = 80)]
        groups: usize,
        #[arg(long, default_value_t = 4)]
        rollouts: usize,
        #[arg(long)]
        test_limit: Option<usize>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Gen { out } => {
            let mut eps = task::generate_all();
            let failures = dronevl::oracle::validate_episodes(&mut eps);
            if !failures.is_empty() {
                anyhow::bail!("oracle failures: {failures:?}");
            }
            task::write_suite(&out, &eps)?;
            println!("wrote {} episodes to {}", eps.len(), out.display());
        }
        Cmd::M1 { uavflow, out, epochs, stride, seed } => {
            let demos = dronevl::uavflow::load(&uavflow)?;
            let samples = dronevl::train::build_sft_samples(&demos, stride);
            let (m1, log) = dronevl::train::train_sft(
                &dronevl::policy::LinearPolicy::m0(),
                &samples,
                epochs,
                64,
                0.01,
                1e-4,
                seed,
            );
            m1.save(&out)?;
            let log_path = out.with_extension("sft.json");
            std::fs::write(log_path, serde_json::to_string_pretty(&log)?)?;
            println!("wrote {} ({} samples, val loss {:.4})", out.display(), samples.len(), log.val_loss.last().copied().unwrap_or(0.0));
        }
        Cmd::Smoke { out, uavflow, epochs, test_limit } => {
            let cfg = ExperimentConfig {
                out_dir: out,
                uavflow_path: uavflow,
                seeds: vec![0],
                epochs,
                sft_stride: 4,
                groups: 8,
                rollouts: 4,
                test_limit: Some(test_limit),
            };
            experiment::smoke(&cfg)?;
        }
        Cmd::Experiment { out, uavflow, seeds, epochs, sft_stride, groups, rollouts, test_limit } => {
            let seeds: Vec<u64> = seeds
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            let cfg = ExperimentConfig {
                out_dir: out,
                uavflow_path: uavflow,
                seeds: if seeds.is_empty() { vec![0] } else { seeds },
                epochs,
                sft_stride,
                groups,
                rollouts,
                test_limit,
            };
            experiment::run(&cfg)?;
        }
    }
    Ok(())
}
