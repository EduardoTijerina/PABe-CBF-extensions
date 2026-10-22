"""Experimentos del banco G1 (MuJoCo). Particiones disjuntas por participante:

  train 0.. / calib 1000.. (train_g1) | validación 3000.. | test 4000.. | sensibilidad 5000.. | controlado 6000..

Uso:  python -m pabe_cbf.g1.experiments {validate,test,sensitivity,controlled,equivalence,all} --out results/g1
Cada etapa guarda <out>/<etapa>.pkl y las posteriores leen los parámetros elegidos en validación.
"""
import argparse
import os
import pickle
import time

import numpy as np

from ..belief import entropy
from ..core import METHODS, method_from_spec
from ..metrics import rate
from ..scenarios import FAMILIES_G1, Participant, Scenario
from .episode import G1Episode
from .geometry import G1_GEOM
from .train_g1 import load

VAL0, TEST0, SENS0, CTRL0 = 3000, 4000, 5000, 6000
SWEEP = {"B3": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
         "B4": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
         "B5": [f"{t}@a{al}" for t in (0.5, 0.7, 0.8, 0.9) for al in (0.1, 0.25, 0.4)],   # tau x alpha (CRC)
         "B8": ["a0.02", "a0.05", "a0.1", "a0.2", "a0.3"],
         "B9": [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98],
         "A6": [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98],
         "A8": [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]}
KAPPAS = ["k1", "k2", "k4", "k8", "k16", "kinf"]
FOLLOW_B9 = ["A1", "A2", "A3", "A4", "A5", "A7", "A9", "A10"]
STRESS = {"noise_scale": [1.0, 2.0, 3.0, 4.0], "vis_delay": [0.0, 0.05, 0.10, 0.15],
          "obs_noise": [1.0, 1.5, 2.0, 3.0], "ext_force": [0.0, 10.0, 20.0, 30.0],
          "tactile_delay": [0.0, 0.1, 0.25, 0.5], "lunge_amp": [0.06, 0.12, 0.18, 0.24]}
_M = None


def _init(models):
    global _M
    _M = models


def _task(t):
    spec, fam, pid, ep, shift, stress, clamp, tag = t
    sc = Scenario(fam, Participant(pid, shift=shift, geom=G1_GEOM), ep, stress=stress)
    res, extra, _ = G1Episode(method_from_spec(spec), sc, _M, clamp=clamp).run()
    res["tag"] = tag
    res["spec"] = spec
    return res, extra


def _tkey(t):
    spec, fam, pid, ep, shift, stress, clamp, tag = t
    return (spec, fam, pid, ep, tag)


def _chunk_worker(models, chunk, path):
    """Proceso hijo: ejecuta un lote y escribe cada resultado en su propio fichero (flujo de pickles)."""
    _init(models)
    with open(path, "ab") as fh:
        for t in chunk:
            pickle.dump(_task(t), fh)
            fh.flush()


def _read_stream(path):
    out = []
    if os.path.exists(path):
        with open(path, "rb") as f:
            while True:
                try:
                    out.append(pickle.load(f))
                except Exception:
                    break
    return out


def run(tasks, models, workers, keep_extra=True, log=print, label="", ckpt=None, chunk=12):
    """Ejecución paralela tolerante a fallos sin pools: lotes de `chunk` tareas, cada uno en un proceso nuevo
    que escribe en su propio fichero <ckpt>.d/<n>.pkl. Si un proceso muere (p. ej. falta de memoria), sus tareas
    no terminadas se reencolan. Al reiniciar, se omiten las tareas ya presentes en los ficheros."""
    import multiprocessing as mp
    from .episode import get_model
    get_model()                                         # crea el MJB antes de lanzar procesos
    ckdir = (ckpt or os.path.join("/tmp", f"pabe_{os.getpid()}_{label}")) + ".d"
    os.makedirs(ckdir, exist_ok=True)
    key = lambda r: (r[0]["spec"], r[0]["family"], r[0]["pid"], r[0]["seed"], r[0]["tag"])
    results = {}
    for fn in sorted(os.listdir(ckdir)):
        for r in _read_stream(os.path.join(ckdir, fn)):
            results[key(r)] = r if keep_extra else (r[0], None)
    if results:
        log(f"[{label}] reanudando: {len(results)} episodios ya hechos")
    todo = [t for t in tasks if _tkey(t) not in results]
    queue = [todo[k:k + chunk] for k in range(0, len(todo), chunk)]
    ctx = mp.get_context("fork")
    running, t0, n_total, n_prev, fails = {}, time.time(), len(tasks), len(results), 0
    seq, last_log = len(os.listdir(ckdir)), len(results)
    while queue or running:
        while queue and len(running) < workers:
            ch = queue.pop(0)
            path = os.path.join(ckdir, f"{seq:06d}.pkl"); seq += 1
            p = ctx.Process(target=_chunk_worker, args=(models, ch, path), daemon=True)
            p.start()
            running[p] = (ch, path)
        time.sleep(0.5)
        for p in list(running):
            if p.is_alive():
                continue
            ch, path = running.pop(p)
            p.join()
            for r in _read_stream(path):
                results[key(r)] = r if keep_extra else (r[0], None)
            if p.exitcode != 0:
                miss = [t for t in ch if _tkey(t) not in results]
                fails += 1
                log(f"[{label}] proceso terminó con código {p.exitcode}; se reencolan {len(miss)} tareas")
                if fails > 50:
                    raise RuntimeError(f"[{label}] demasiados fallos de procesos")
                if miss:
                    queue.append(miss)
        if len(results) - last_log >= 500:
            last_log = len(results)
            el = time.time() - t0
            k = len(results) - n_prev
            log(f"[{label}] {len(results)}/{n_total} episodios, {el / 60:.1f} min, "
                f"ETA {(n_total - len(results)) * el / max(k, 1) / 60:.1f} min")
    return [results[_tkey(t)] for t in tasks]


def tasks_for(specs, pid0, n_parts, eps, families=FAMILIES_G1, shift_frac=0.25, stress=None, clamp=None, tag=""):
    n_shift = int(round(shift_frac * n_parts))
    return [(s, f, pid0 + p, e, p < n_shift, stress, clamp, tag) for s in specs for f in families
            for p in range(n_parts) for e in range(eps)]


def detected(r):
    if not r["withdraw"] or r["t_R"] is None or r["R_reason"] not in ("sem", "abort"):
        return False
    return r["t_W"] <= r["t_R"] <= r["t_W"] + 1.0


def by_spec(out):
    rows = {}
    for r, _ in out:
        rows.setdefault(r["spec"], []).append(r)
    return rows


# --------------------------------------------------------------------------------------
def validate(models, a, log):
    specs = [f"{b}@{t}" for b, ts in SWEEP.items() for t in ts] + ["B2"]
    out = run(tasks_for(specs, VAL0, a.val_parts, a.val_eps), models, a.workers, False, log, "validación-umbral",
              ckpt=os.path.join(a.out, "validate_thr.ckpt"))
    rows = by_spec(out)
    front = {}
    for s, rs in rows.items():
        b = s.split("@")[0]
        front.setdefault(b, []).append(dict(spec=s, sens=rate([detected(r) for r in rs if r["withdraw"]]),
                                            fc=rate([r["false_cancel"] for r in rs]),
                                            inf=rate([r["n_infeas"] > 0 for r in rs]),
                                            jerk=float(np.mean([r["jerk_p95"] for r in rs]))))
    # objetivo común fijo; quien no lo alcance usa su punto de máxima sensibilidad (se reporta)
    s_star = a.target
    unreached = [b for b, pts in front.items() if b in SWEEP and max(p["sens"]["rate"] for p in pts) < s_star]
    log(f"[validación] métodos que no alcanzan s*={s_star}: {unreached}")
    chosen = {}
    for b, pts in front.items():
        if b not in SWEEP:
            continue
        ok = [p for p in pts if p["sens"]["rate"] >= s_star - 1e-9] or [max(pts, key=lambda p: p["sens"]["rate"])]
        chosen[b] = min(ok, key=lambda p: (p["fc"]["rate"], -p["sens"]["rate"]))["spec"]
    log(f"[validación] s*={s_star:.3f}; elegidos: {chosen}")
    # mejor ley exógena para B6 (con el umbral de B9): mínima infactibilidad y, a igualdad (<=1 pp), menor jerk
    tau9 = chosen["B9"].split("@")[1]
    kspecs = [f"B6@{tau9}@{k}" for k in KAPPAS]
    out2 = run(tasks_for(kspecs, VAL0, a.val_parts, a.val_eps), models, a.workers, False, log, "validación-kappa",
               ckpt=os.path.join(a.out, "validate_kappa.ckpt"))
    rows2 = by_spec(out2)
    kres = {s: dict(inf=float(np.mean([r["n_infeas"] > 0 for r in rs])), jerk=float(np.mean([r["jerk_p95"] for r in rs])),
                    fc=float(np.nanmean([r["false_cancel"] for r in rs])), viol=float(np.mean([r["V_any"] for r in rs])))
            for s, rs in rows2.items()}
    best_inf = min(v["inf"] for v in kres.values())
    cand = {s: v for s, v in kres.items() if v["inf"] <= best_inf + 0.01}
    chosen["B6"] = min(cand, key=lambda s: cand[s]["jerk"])
    log(f"[validación] B6*: {chosen['B6']}  ({kres})")
    res = dict(front=front, s_star=s_star, chosen=chosen, kappa=kres, rows=rows, rows_kappa=rows2, unreached=unreached)
    with open(os.path.join(a.out, "validate.pkl"), "wb") as f:
        pickle.dump(res, f)
    return res


def main_specs(chosen):
    tau9 = chosen["B9"].split("@")[1]
    specs = ["B0", "B1", "B2", chosen["B3"], chosen["B4"], chosen["B5"], chosen["B6"], "B7", chosen["B8"],
             chosen["B9"], chosen["A6"], chosen["A8"]] + [f"{m}@{tau9}" for m in FOLLOW_B9]
    extra = [f"B6@{tau9}@{k}" for k in KAPPAS]          # barrido de kappa (atribución) + B6 con kappa de referencia
    return specs, [s for s in extra if s not in specs]


def test(models, a, log, V):
    specs, extra = main_specs(V["chosen"])
    out = run(tasks_for(specs + extra, TEST0, a.test_parts, a.test_eps), models, a.workers, True, log, "test",
              ckpt=os.path.join(a.out, "test.ckpt"))
    with open(os.path.join(a.out, "test.pkl"), "wb") as f:
        pickle.dump(dict(out=out, specs=specs, extra=extra, chosen=V["chosen"]), f)


def sensitivity(models, a, log, V):
    c = V["chosen"]
    tau9 = c["B9"].split("@")[1]
    specs = ["B0", "B1", "B2", c["B3"], c["B4"], c["B5"], c["B6"], "B7", c["B8"], c["B9"], f"A1@{tau9}"]
    tasks = []
    for fac, levels in STRESS.items():
        for lv in levels:
            fams = ("vii_lunge",) if fac == "lunge_amp" else FAMILIES_G1
            eps = 3 if fac == "lunge_amp" else 1
            tasks += tasks_for(specs, SENS0, a.sens_parts, eps, fams, stress={fac: lv}, tag=f"{fac}={lv}")
    out = run(tasks, models, a.workers, False, log, "sensibilidad", ckpt=os.path.join(a.out, "sensitivity.ckpt"))
    with open(os.path.join(a.out, "sensitivity.pkl"), "wb") as f:
        pickle.dump(dict(rows=[r for r, _ in out], specs=specs), f)


def controlled_cells():
    """Rejilla (p_W, a): b = ((1-p)(1-a), (1-p)a, p). Con p fijo, a sube la entropía.
    Más celdas iso-entropía: H fijo y p variable (Prop. 1)."""
    cells = []
    for p in (0.02, 0.10, 0.20, 0.35, 0.50):
        for aa in (0.0, 0.15, 0.30, 0.50):
            b = np.array([(1 - p) * (1 - aa), (1 - p) * aa, p])
            cells.append(dict(kind="grid", p=p, a=aa, b=b, H=entropy(b)))
    for H0 in (0.40, 0.60):
        for p in (0.02, 0.10, 0.20, 0.35, 0.50):
            xs = np.linspace(0, 0.5, 2001)
            bs = [np.array([(1 - p) * (1 - x), (1 - p) * x, p]) for x in xs]
            Hs = np.array([entropy(b) for b in bs])
            i = int(np.argmin(abs(Hs - H0)))
            if abs(Hs[i] - H0) < 0.01:
                cells.append(dict(kind=f"iso{H0}", p=p, a=float(xs[i]), b=bs[i], H=float(Hs[i])))
    return cells


def controlled(models, a, log, V):
    c = V["chosen"]
    tau9 = c["B9"].split("@")[1]
    specs = ["B1", "B2", c["B3"], c["B4"], c["B5"], c["B6"], c["B8"], c["B9"], f"A2@{tau9}"]
    tasks = []
    for i, cell in enumerate(controlled_cells()):
        clamp = dict(start=0.3, dur=0.8, b=cell["b"].tolist())
        tasks += tasks_for(specs, CTRL0, a.ctrl_parts, 1, ("i_stable",), shift_frac=0.0, clamp=clamp, tag=str(i))
    out = run(tasks, models, a.workers, False, log, "controlado", ckpt=os.path.join(a.out, "controlled.ckpt"))
    with open(os.path.join(a.out, "controlled.pkl"), "wb") as f:
        pickle.dump(dict(rows=[r for r, _ in out], cells=controlled_cells(), specs=specs), f)


def equivalence(models, a, log, V):
    """A9 (gobernador con rho->inf) frente a B6 con la misma ley de referencia (kappa=4): mientras el gobernador
    no se activa, ambos deben producir exactamente la misma trayectoria (mismo código, mismo QP)."""
    tau9 = V["chosen"]["B9"].split("@")[1]
    res = []
    for fam in FAMILIES_G1:
        for pid in range(TEST0, TEST0 + 3):
            recs = {}
            for spec in (f"A9@{tau9}", f"B6@{tau9}@k4"):
                ep = G1Episode(method_from_spec(spec), Scenario(fam, Participant(pid, geom=G1_GEOM), 0), models,
                               record=True)
                ep.run()
                recs[spec] = ep.record
            ra, rb = recs[f"A9@{tau9}"], recs[f"B6@{tau9}@k4"]
            n = min(len(ra), len(rb))
            du = np.array([np.abs(ra[i]["u"] - rb[i]["u"]).max() for i in range(n)])
            first = next((i for i in range(n) if du[i] > 1e-4), None)
            g0 = next((i for i in range(n) if ra[i]["gov"]), None)       # primera activación del gobernador en A9
            res.append(dict(fam=fam, pid=pid, n=n, first_div=(ra[first]["t"] if first is not None else None),
                            max_du_before=float(du[:first].max()) if first else float(du.max()),
                            first_gov=(ra[g0]["t"] if g0 is not None else None),
                            div_after_gov=(first is None) or (g0 is not None and g0 <= first + 1),
                            b6_infeasible_at_div=(first is not None and not rb[first]["feasible"]),
                            explained=(first is None) or (g0 is not None and g0 <= first + 1) or (not rb[first]["feasible"]),
                            feasible_A9_at_div=(ra[first]["feasible"] if first is not None else None)))
    with open(os.path.join(a.out, "equivalence.pkl"), "wb") as f:
        pickle.dump(res, f)
    log(f"[equivalencia] {sum(r['first_div'] is None for r in res)}/{len(res)} idénticos de principio a fin; "
        f"{sum(r['explained'] for r in res)}/{len(res)} divergen sólo cuando la ley exógena es inviable o el gobernador actúa")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["validate", "test", "sensitivity", "controlled", "equivalence", "all"])
    ap.add_argument("--out", default="results/g1")
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--val-parts", type=int, default=6)
    ap.add_argument("--val-eps", type=int, default=2)
    ap.add_argument("--test-parts", type=int, default=20)
    ap.add_argument("--test-eps", type=int, default=2)
    ap.add_argument("--sens-parts", type=int, default=4)
    ap.add_argument("--ctrl-parts", type=int, default=10)
    ap.add_argument("--target", type=float, default=0.8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "experiments.log"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()
    models = load(os.path.join(a.out, "models.pkl"))
    stages = ["validate", "test", "sensitivity", "controlled", "equivalence"] if a.stage == "all" else [a.stage]
    V = None
    vpath = os.path.join(a.out, "validate.pkl")
    for st in stages:
        t0 = time.time()
        if st == "validate":
            V = validate(models, a, log)
            continue
        if V is None:
            with open(vpath, "rb") as f:
                V = pickle.load(f)
        dict(test=test, sensitivity=sensitivity, controlled=controlled, equivalence=equivalence)[st](models, a, log, V)
        log(f"[{st}] terminado en {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
