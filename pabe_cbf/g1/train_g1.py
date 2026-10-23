"""Entrenamiento/calibración del banco G1 con particiones disjuntas por participante.

  train 0..   : HMM (MLE supervisado, emisiones t), HSMM (A6), H_ref, cota cartesiana w de seguimiento
  calib 1000..: p_W^+ isotónica (HMM y HSMM), cuantil CRC (B5), puntuaciones RCIP (B8)
Los datos se recogen con B1 (CBF fijo, sin HMM) en las nueve familias G1.

Sistema de aproximación (models.approach, train_approach): HMM propio sobre y_a = (y[1], y[3]) con la fase de
caminata de FAMILIES_WALK (política ciega P0, episodios que terminan en el traspaso); en calibración, p_W⁺
isotónica y H_ref (cuantil 0,8 de la entropía con la persona dispuesta).
"""
import os
import pickle
from multiprocessing import Pool

import numpy as np

from ..approach import fit_hmm as fit_approach_hmm
from ..belief import ModeHMM, ModeHSMM, HRef, UpperCalibrator, IsoUpperCalibrator, conformal_quantile
from ..core import METHODS, Models
from ..scenarios import FAMILIES_G1, FAMILIES_WALK, Participant, Scenario
from .episode import G1Episode
from .geometry import G1_GEOM

_M = None


def _init(models):
    global _M
    _M = models


def _collect(task):
    fam, pid, ep, hsmm = task
    m = METHODS["A6"] if hsmm else METHODS["B1"]
    from dataclasses import replace
    # recolección con la regla B1 (sin decisiones semánticas); A6 sólo para obtener creencias del HSMM
    m = replace(m, risk="none", theta_update="fixed", prudence=False) if hsmm else m
    return G1Episode(m, Scenario(fam, Participant(pid, geom=G1_GEOM), ep), _M, collect=True).run()[2]


def _collect_approach(task):
    fam, pid, ep = task
    return G1Episode(METHODS["B1"], Scenario(fam, Participant(pid, geom=G1_GEOM), ep), _M, collect=True,
                     approach="P0", approach_only=True).run()[2]


def _pass(tasks, models, workers, fn=_collect):
    with Pool(workers, initializer=_init, initargs=(models,), maxtasksperchild=50) as p:
        return p.map(fn, tasks, chunksize=2)


def h_durations(recs):
    out, run = [], 0
    for g, z, y, zp in recs:
        if z == 1:
            run += 1
        elif run:
            out.append(run); run = 0
    return out


def train(n_train=12, n_cal=10, eps=3, workers=10, alpha=0.1, log=print):
    tr = [(f, p, e) for f in FAMILIES_G1 for p in range(n_train) for e in range(eps)]
    ca = [(f, 1000 + p, e) for f in FAMILIES_G1 for p in range(n_cal) for e in range(eps)]
    out = _pass([t + (False,) for t in tr], None, workers)
    recs = [r for c in out for r in c["obs"]]
    hmm = ModeHMM.fit(recs)
    # identificación en operación nominal: la familia de perturbación excede la hipótesis A2 a propósito
    trk = np.concatenate([np.asarray(c["track"]) for c, t in zip(out, tr) if t[0] != "vi_perturbation"])
    log(f"[train-g1] error cartesiano de seguimiento p99/p99,5/p99,9 = {np.percentile(trk, [99, 99.5, 99.9]).round(3)}")
    w_cart = float(np.percentile(trk[np.isfinite(trk)], 99.9))
    hsmm = ModeHSMM.from_hmm(hmm, h_durations(recs))
    models = Models(hmm.A, hmm.mu, hmm.sd, hmm.prior, href=None, calib=None, crc_q=None, rcip_q=None, w_design=w_cart)
    models.hsmm = dict(k=hsmm.k, A_exp=hsmm.A_exp, calib=None, href=None)
    log(f"[train-g1] HMM con {len(recs)} obs.; HSMM k={hsmm.k}; w_cart(p99,9)={w_cart:.3f} m/s")
    # H_ref (HMM y HSMM) en trazas nominales
    for hs in (False, True):
        o = _pass([t + (hs,) for t in tr], models, workers)
        Hs = [(g, tau, H) for c in o for (g, tau, H, fam, nominal) in c["H"] if nominal and fam in ("i_stable", "iv_hesitation")]
        if hs:
            models.hsmm["href"] = HRef.fit(Hs, q=0.8)
        else:
            models.href = HRef.fit(Hs, q=0.8)
    # calibración
    for hs in (False, True):
        o = _pass([t + (hs,) for t in ca], models, workers)
        p = [a for c in o for a, _ in c["cal"]]
        y = [b for c in o for _, b in c["cal"]]
        epi = [i for i, c in enumerate(o) for _ in c["cal"]]
        iso = IsoUpperCalibrator.fit(p, y, epi)
        if hs:
            models.hsmm["calib"] = iso
        else:
            models.calib = iso
            models.calib_bins = UpperCalibrator.fit(p, y)
            crc = [r for c in o for r in c["crc"]]
            # cuantil conforme contextual: grupo de modo x velocidad de la mano (lenta/rápida)
            models.crc_q = np.array([[conformal_quantile([e for g, e, v in crc if g == gg and v == vv] or [0.1], alpha)
                                      for vv in (0, 1)] for gg in (0, 1)])
            # errores crudos por contexto: permiten elegir el nivel alpha de B5 en validación
            models.crc_err = {(gg, vv): np.array([e for g, e, v in crc if g == gg and v == vv] or [0.1])
                              for gg in (0, 1) for vv in (0, 1)}
            models.rcip_scores = np.array([s for c in o for s in c["rcip"]])
            models.rcip_q = conformal_quantile(models.rcip_scores, alpha)
            log(f"[train-g1] calibración: {len(p)} ticks (tasa W={np.mean(y):.3f}); q_CRC={models.crc_q.round(3)}; "
                f"q_RCIP={models.rcip_q:.3f}")
    models.approach = train_approach(n_train, n_cal, 2 * eps, workers, log)
    return models


def train_approach(n_train, n_cal, eps, workers, log=print):
    """Sistema de aproximación: HMM sobre la fase de caminata, luego p_W⁺ calibrada y H_ref.
    eps: episodios por participante (las seis variantes de la caminata se sortean por episodio)."""
    tr = [(f, p, e) for f in FAMILIES_WALK for p in range(n_train) for e in range(eps)]
    ca = [(f, 1000 + p, e) for f in FAMILIES_WALK for p in range(n_cal) for e in range(eps)]
    out = _pass(tr, None, workers, _collect_approach)
    am = fit_approach_hmm([r for c in out for r in c["obs_a"]])
    log(f"[train-approach] HMM con {sum(len(c['obs_a']) for c in out)} obs.; y[3] por E/H/W: {am.mu[0, :, 1].round(2)}")
    holder = Models(None, None, None, None, href=None, calib=None, crc_q=None, rcip_q=None, w_design=None)
    holder.approach = am
    o = _pass(ca, holder, workers, _collect_approach)
    p = [a for c in o for a, _ in c["cal_a"]]
    y = [b for c in o for _, b in c["cal_a"]]
    epi = [i for i, c in enumerate(o) for _ in c["cal_a"]]
    am.calib = IsoUpperCalibrator.fit(p, y, epi)
    am.href = float(np.quantile([h for c in o for h, z in c["H_a"] if z == 0], 0.8))
    log(f"[train-approach] calibración: {len(p)} ticks (tasa W={np.mean(y):.3f}); H_ref={am.href:.3f}")
    return am


def save(models, path):
    with open(path, "wb") as f:
        pickle.dump(models, f)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)
