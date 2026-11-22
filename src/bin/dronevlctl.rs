use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use dronevl::{cache, eval, experiment_plan, orchestrator, proxy, unrealcv};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser)]
#[command(name = "dronevlctl", about = "DroneVL experiment control plane")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Summarize one eval results directory and write a JSON report.
    EvalSummary {
        #[arg(long)]
        dir: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Paired comparison of two eval result directories.
    EvalCompare {
        #[arg(long)]
        baseline: PathBuf,
        #[arg(long)]
        candidate: PathBuf,
        #[arg(long, default_value = "")]
        out: PathBuf,
    },
    /// Write the experiment 1–8 plan as CSV and JSON.
    Plan {
        #[arg(long, default_value = "runs/plan")]
        out_dir: PathBuf,
    },
    /// Run the HTTP policy gateway in front of a Python OpenVLA server.
    Proxy {
        #[arg(long, default_value = "127.0.0.1:5007")]
        bind: SocketAddr,
        #[arg(long, default_value = "http://127.0.0.1:8010/predict")]
        upstream: String,
        #[arg(long, default_value_t = 120)]
        timeout_s: u64,
    },
    /// Build a content-addressed expert cache.
    Cache {
        #[arg(long)]
        src: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Run one supervised command with timeout/retries.
    Run {
        #[arg(long, default_value_t = 3600)]
        timeout_s: u64,
        #[arg(long, default_value_t = 0)]
        retries: u32,
        #[arg(required = true, num_args = 1.., last = true)]
        command: Vec<String>,
    },
    /// Send one command to an UnrealCV socket (smoke test).
    UnrealcvProbe {
        #[arg(long)]
        addr: SocketAddr,
        #[arg(long, default_value = "vget /camera/0/lit")]
        command: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::EvalSummary { dir, out } => {
            let r = eval::write_report(&dir, &out)?;
            println!("{}", serde_json::to_string_pretty(&r.results)?);
        }
        Cmd::EvalCompare {
            baseline,
            candidate,
            out,
        } => {
            let r = eval::compare_dirs(&baseline, &candidate)?;
            let text = serde_json::to_string_pretty(&r)?;
            if !out.as_os_str().is_empty() {
                std::fs::write(&out, &text)?;
            }
            println!("{text}");
        }
        Cmd::Plan { out_dir } => {
            std::fs::create_dir_all(&out_dir)?;
            experiment_plan::write_csv(&out_dir.join("experiment_plan.csv"))?;
            experiment_plan::write_json(&out_dir.join("experiment_plan.json"))?;
            let n = experiment_plan::plan().len();
            println!("wrote {} runs to {}", n, out_dir.display());
        }
        Cmd::Proxy {
            bind,
            upstream,
            timeout_s,
        } => {
            let cfg = proxy::ProxyConfig {
                bind,
                upstream,
                timeout: Duration::from_secs(timeout_s),
            };
            let server = proxy::ProxyServer::bind(cfg)?;
            println!("proxy listening on {}", server.local_addr()?);
            server.serve_forever()?;
        }
        Cmd::Cache { src, out } => {
            let m = cache::build(&src, &out)?;
            println!("cached {} files, {} bytes", m.files, m.total_bytes);
        }
        Cmd::Run {
            timeout_s,
            retries,
            command,
        } => {
            let mut it = command.into_iter();
            let program = it.next().context("missing program")?;
            let mut spec = orchestrator::RunSpec::new(program)
                .timeout(timeout_s)
                .retries(retries);
            spec.args = it.collect();
            let out = orchestrator::run(&spec)?;
            println!("{}", serde_json::to_string_pretty(&out)?);
            if !out.success {
                std::process::exit(1);
            }
        }
        Cmd::UnrealcvProbe { addr, command } => {
            let mut c = unrealcv::UnrealCVClient::connect(addr, Duration::from_secs(5))?;
            println!("{}", c.command(&command)?);
        }
    }
    Ok(())
}
