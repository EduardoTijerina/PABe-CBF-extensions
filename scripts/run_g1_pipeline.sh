#!/usr/bin/env bash
# Entrenamiento + validación + test + sensibilidad + experimento controlado + equivalencia (G1, MuJoCo).
# Tolerante a fallos: si se interrumpe, al relanzarlo retoma desde results/g1/*.ckpt.d (~90 min con 10 procesos).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
PY="${PYTHON:-python}"
WORKERS="${WORKERS:-10}"
mkdir -p results/g1
if [ ! -f results/g1/models.pkl ]; then
  "$PY" - <<PY
from pabe_cbf.g1.train_g1 import train, save
m = train(n_train=12, n_cal=10, eps=3, workers=$WORKERS, log=lambda s: print(s, flush=True))
save(m, "results/g1/models.pkl")
PY
fi
"$PY" -m pabe_cbf.g1.experiments all --out results/g1 --workers "$WORKERS"
"$PY" scripts/run_a11.py --out results/g1 --workers "$WORKERS"
"$PY" -m pabe_cbf.g1.report --out results/g1
"$PY" -m pabe_cbf.g1.videos --out results/g1
