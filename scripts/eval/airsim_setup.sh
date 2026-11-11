#!/usr/bin/env bash
# AirSimNH 1.8.1 offscreen setup for a Linux GPU pod (OpenGL fallback).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq unzip libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 python3-venv >/dev/null 2>&1
python3 -m venv /workspace/venv-airsim
/workspace/venv-airsim/bin/pip install -q numpy pillow setuptools wheel msgpack-rpc-python msgpack==0.6.2 backports.ssl_match_hostname
/workspace/venv-airsim/bin/pip install -q --no-build-isolation airsim
mkdir -p /workspace/airsim && cd /workspace/airsim
[ -f AirSimNH.zip ] || curl -L --retry 3 -o AirSimNH.zip https://github.com/microsoft/AirSim/releases/download/v1.8.1/AirSimNH.zip
[ -d AirSimNH ] || unzip -q AirSimNH.zip
BIN=$(find /workspace/airsim -type f -name AirSimNH | head -1)
id ue4 >/dev/null 2>&1 || useradd -m ue4
usermod -aG video,render ue4 2>/dev/null || true
chown -R ue4:ue4 /workspace/airsim
runuser -u ue4 -- bash -c "cd /workspace/airsim/AirSimNH/LinuxNoEditor && nohup $BIN -RenderOffScreen -opengl -ResX=1280 -ResY=720 -unattended > /home/ue4/airsim.log 2>&1 &"
for _ in $(seq 1 30); do ss -ltn | grep -q 41451 && exit 0; sleep 2; done
echo "AirSimNH RPC did not start"; exit 1
