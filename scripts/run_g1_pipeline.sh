#!/usr/bin/env bash
# Entrenamiento + validación + test + sensibilidad + experimento controlado + equivalencia (sistema de saludo),
# banco del sistema de aproximación y encadenado caminata -> saludo (G1, MuJoCo).
# Tolerante a fallos: si se interrumpe, al relanzarlo retoma desde $OUT/*.ckpt.d (~90 min con 10 procesos).
# Robot: PABE_ROBOT=g1 (por defecto, Menagerie + Dex3) o PABE_ROBOT=revo2 (URDF 51 GDL + Revo2); cada uno escribe
# en su propio directorio (results/g1, results/g1_revo2) salvo que se fije OUT.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
PY="${PYTHON:-python}"
WORKERS="${WORKERS:-10}"
OUT="${OUT:-$("$PY" -c 'from pabe_cbf.g1.model import RESULTS_DIR; print(RESULTS_DIR)')}"
"$PY" -c "from pabe_cbf.g1.model import check_results_dir; check_results_dir('$OUT')"
echo "robot=${PABE_ROBOT:-g1} out=$OUT"
if [ ! -f "$OUT/models.pkl" ]; then
  "$PY" - <<PY
from pabe_cbf.g1.train_g1 import train, save
m = train(n_train=12, n_cal=10, eps=3, workers=$WORKERS, log=lambda s: print(s, flush=True))
save(m, "$OUT/models.pkl")
PY
fi
"$PY" -m pabe_cbf.g1.experiments all --out "$OUT" --workers "$WORKERS"
"$PY" scripts/run_a11.py --out "$OUT" --workers "$WORKERS"
"$PY" -m pabe_cbf.g1.experiments approach --out "$OUT" --workers "$WORKERS"   # banco A: sistema de aproximación
"$PY" -m pabe_cbf.g1.experiments walk --out "$OUT" --workers "$WORKERS"       # encadenado: aproximación P3 + saludo
"$PY" -m pabe_cbf.g1.report --out "$OUT"
"$PY" -m pabe_cbf.g1.videos --out "$OUT"
