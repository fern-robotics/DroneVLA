//! Expert-cache and dataset materialization.
//!
//! Copies a source tree (frames, JSONL, episode logs) into a content-addressed
//! cache and writes a manifest. This is the Rust replacement for the slow
//! Python copy/hash loops; training still reads the same JSONL format.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheFile {
    pub source: String,
    pub dest: String,
    pub bytes: u64,
    pub hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheManifest {
    pub files: usize,
    pub total_bytes: u64,
    pub entries: Vec<CacheFile>,
}

fn hash_bytes(b: &[u8]) -> String {
    let mut h = DefaultHasher::new();
    b.hash(&mut h);
    format!("{:016x}", h.finish())
}

fn walk(root: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(root).with_context(|| format!("read {}", root.display()))? {
        let p = entry?.path();
        if p.is_dir() {
            walk(&p, out)?;
        } else if p.is_file() {
            out.push(p);
        }
    }
    Ok(())
}

pub fn build(src: &Path, out: &Path) -> Result<CacheManifest> {
    let mut files = Vec::new();
    walk(src, &mut files)?;
    files.sort();
    let mut entries = Vec::new();
    let mut total = 0u64;
    for f in files {
        let rel = f.strip_prefix(src).unwrap_or(&f).to_path_buf();
        let dest = out.join(&rel);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let bytes = std::fs::read(&f)?;
        std::fs::write(&dest, &bytes)?;
        total += bytes.len() as u64;
        entries.push(CacheFile {
            source: f.display().to_string(),
            dest: dest.display().to_string(),
            bytes: bytes.len() as u64,
            hash: hash_bytes(&bytes),
        });
    }
    let manifest = CacheManifest {
        files: entries.len(),
        total_bytes: total,
        entries,
    };
    std::fs::write(
        out.join("cache_manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn copies_and_manifests() {
        let root = std::env::temp_dir().join(format!("dronevl-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let src = root.join("src");
        let out = root.join("out");
        std::fs::create_dir_all(src.join("frames")).unwrap();
        std::fs::write(src.join("episode.jsonl"), b"{\"a\":1}\n").unwrap();
        std::fs::write(src.join("frames/0001.jpg"), b"img").unwrap();
        let m = build(&src, &out).unwrap();
        assert_eq!(m.files, 2);
        assert!(out.join("cache_manifest.json").exists());
        assert!(out.join("frames/0001.jpg").exists());
    }
}
