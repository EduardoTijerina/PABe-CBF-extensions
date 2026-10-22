#!/usr/bin/env bash
# Descarga el modelo Unitree G1 de MuJoCo Menagerie (licencia BSD-3 de Unitree; no se redistribuye aquí).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/third_party"
cd "$ROOT/third_party"
if [ ! -d mujoco_menagerie ]; then
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
  git -C mujoco_menagerie sparse-checkout set unitree_g1
fi
echo "Modelo G1 en third_party/mujoco_menagerie/unitree_g1"
