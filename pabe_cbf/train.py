"""Ajuste del HMM, H_ref y calibradores con particiones disjuntas por participante (Sec. 7.5).

  train  (pid 0..)    -> parámetros HMM (MLE supervisado) y H_ref(m, tau) en trazas nominales
  calib  (pid 1000..) -> p_W^+ (Clopper-Pearson por bin), cuantil CRC (B5), cuantil RCIP (B8)
  test   (pid 2000..) -> sólo evaluación (run.py)
Los datos se recogen con el controlador B1 (CBF fijo), independiente del HMM.
"""
import pickle
from multiprocessing import Pool

import numpy as np

from .belief import ModeHMM, HRef, UpperCalibrator, IsoUpperCalibrator, conformal_quantile
from .scenarios import FAMILIES, Participant, Scenario
from .sim import Episode, METHODS, Models

TRAIN_PID0, CAL_PID0, TEST_PID0 = 0, 1000, 2000
_MODELS = None


def _init(models):
    global _MODELS
    _MODELS = models


def _collect(task):
    fam, pid, ep = task
    scen = Scenario(fam, Participant(pid), ep)
    return Episode(METHODS["B1"], scen, _MODELS, collect=True).run()[2]


def _run_pass(tasks, models, workers):
    with Pool(workers, initializer=_init, initargs=(models,)) as pool:
        return pool.map(_collect, tasks, chunksize=4)


def train_models(n_train=12, n_cal=10, eps=4, workers=8, alpha=0.1, verbose=True):
    tr = [(f, TRAIN_PID0 + p, e) for f in FAMILIES for p in range(n_train) for e in range(eps)]
    ca = [(f, CAL_PID0 + p, e) for f in FAMILIES for p in range(n_cal) for e in range(eps)]

    # 1) HMM supervisado
    out = _run_pass(tr, None, workers)
    recs = [r for c in out for r in c["obs"]]
    hmm = ModeHMM.fit(recs)
    models = Models(hmm.A, hmm.mu, hmm.sd, hmm.prior, href=None, calib=None, crc_q=None, rcip_q=None)
    if verbose:
        print(f"[train] HMM ajustado con {len(recs)} observaciones")

    # 2) H_ref en trazas nominales (sin retirada; familias i y iv)
    out = _run_pass(tr, models, workers)
    Hs = [(g, tau, H) for c in out for (g, tau, H, fam, nominal) in c["H"]
          if nominal and fam in ("i_stable", "iv_hesitation")]
    models.href = HRef.fit(Hs, q=0.8)

    # 3) calibración en participantes disjuntos
    out = _run_pass(ca, models, workers)
    p = [a for c in out for a, _ in c["cal"]]
    y = [b for c in out for _, b in c["cal"]]
    epi = [i for i, c in enumerate(out) for _ in c["cal"]]
    models.calib_bins = UpperCalibrator.fit(p, y)          # v1 (ablación A8)
    models.calib = IsoUpperCalibrator.fit(p, y, epi)       # v2 (por defecto)
    crc = [r for c in out for r in c["crc"]]
    models.crc_q = np.array([conformal_quantile([e for g, e in crc if g == gg], alpha) for gg in (0, 1)])
    models.rcip_q = conformal_quantile([s for c in out for s in c["rcip"]], alpha)
    if verbose:
        print(f"[train] H_ref con {len(Hs)} muestras; calibración con {len(p)} ticks "
              f"(tasa W={np.mean(y):.3f}); q_CRC={models.crc_q.round(3)}, q_RCIP={models.rcip_q:.3f}")
        print("[train] p_W^+ isotónica (centro, cota):",
              [(round(a, 3), round(b, 3)) for a, b in zip(models.calib.x, models.calib.upper_c)])
    return models


def save(models, path):
    with open(path, "wb") as f:
        pickle.dump(models, f)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)
