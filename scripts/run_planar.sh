#!/usr/bin/env bash
# Banco planar (2D): prueba rápida y evaluación completa v2 con figuras y tablas.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
"$PY" -m pabe_cbf.run --test-parts 8 --eps 2 --out results/quick
"$PY" -m pabe_cbf.paper_exp --out results/paper
"$PY" -m pabe_cbf.paper_figs --out results/paper
