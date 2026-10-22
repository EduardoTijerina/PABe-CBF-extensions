"""Benchmark B0-B9 + ablaciones con semillas apareadas (Sec. 7).

Uso:
  python -m pabe_cbf.run --test-parts 8 --eps 3 --workers 14 --out results/quick
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from . import params as P
from .metrics import summarize, paired_bootstrap
from .scenarios import FAMILIES, Participant, Scenario
from .sim import Episode, METHODS, method_from_spec
from .train import train_models, save, load, TEST_PID0

_MODELS = None
PRIMARY = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"]
ABLATIONS = ["A1", "A2", "A3", "A4", "A5", "A7"]


def _init(models):
    global _MODELS
    _MODELS = models


def _task(args):
    mname, fam, pid, ep, shift = args
    scen = Scenario(fam, Participant(pid, shift=shift), ep)
    res, extra, _ = Episode(method_from_spec(mname), scen, _MODELS).run()
    return res, extra


def fmt_rate(r):
    if r["n"] == 0:
        return "–"
    s = f"{100 * r['rate']:.1f} [{100 * r['lo']:.1f}, {100 * r['hi']:.1f}]"
    if r["k"] == 0:
        s += f" (≤{100 * r['upper0']:.1f})"
    return s


def report(summ, per_method_rows, methods, out):
    L = []
    L.append("# PABe-CBF — prueba rápida del banco de evaluación\n")
    L.append("Simulación planar simplificada (no Isaac Sim, no G1). Tasas con IC 95 % de Clopper–Pearson; "
             "entre paréntesis, cota superior unilateral 95 % cuando hay 0 eventos (Ec. 16).\n")
    L.append("## Seguridad física (resultados primarios)\n")
    L.append("| Método | Violación física % | Dist. eslabón % | Fuerza > F_phys % | Contacto no autorizado % | Pico F medio (N) | Impulso sobre límite (N·s) |")
    L.append("|---|---|---|---|---|---|---|")
    for m in methods:
        s = summ[m]
        L.append(f"| {m} {method_from_spec(m).label} | {fmt_rate(s['V_any'])} | {fmt_rate(s['V_dist_link'])} | "
                 f"{fmt_rate(s['V_force'])} | {fmt_rate(s['V_unauth_contact'])} | {s['peak_F_mean']:.1f} | "
                 f"{s['impulse_over_mean']:.4f} |")
    L.append("\n## Retirada, tarea y factibilidad\n")
    L.append("| Método | Liberación mediana / p95 (ms) | Liberación fallida % | Cancelación falsa % | Éxito autorizado % | Sin contacto (no autoriz.) % | QP infactible (ciclos %) | Episodios con infactib. % |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m in methods:
        s = summ[m]
        L.append(f"| {m} | {s['release_ms_median']:.0f} / {s['release_ms_p95']:.0f} (n={s['release_n']}) | "
                 f"{fmt_rate(s['release_fail'])} | {fmt_rate(s['false_cancel'])} | {fmt_rate(s['success_auth'])} | "
                 f"{fmt_rate(s['nocontact_unauth'])} | {100 * s['infeas_cycles']:.2f} | {fmt_rate(s['infeas_episodes'])} |"
                 if s['infeas_cycles'] == s['infeas_cycles'] else
                 f"| {m} | {s['release_ms_median']:.0f} / {s['release_ms_p95']:.0f} (n={s['release_n']}) | "
                 f"{fmt_rate(s['release_fail'])} | {fmt_rate(s['false_cancel'])} | {fmt_rate(s['success_auth'])} | "
                 f"{fmt_rate(s['nocontact_unauth'])} | – | – |")
    L.append("\n## Intervención, suavidad y tiempo real (eficiencia)\n")
    L.append("| Método | J_int | Jerk palma p95 (m/s³) | |dF/dt| p95 (N/s) | Capa seguridad p50/p95/p99/máx (ms) | QP p50/p99 (ms) | HMM p50 (ms) | % > 150 ms |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m in methods:
        s = summ[m]
        L.append(f"| {m} | {s['J_int_mean']:.3f} | {s['jerk_p95_mean']:.1f} | {s['Fdot_p95_mean']:.1f} | "
                 f"{s['lat_p50']:.3f} / {s['lat_p95']:.3f} / {s['lat_p99']:.3f} / {s['lat_max']:.2f} | "
                 f"{s['qp_p50']:.3f} / {s['qp_p99']:.3f} | {s['hmm_p50']:.3f} | {100 * s['lat_over_budget']:.2f} |")
    L.append("\n## Predicción de retirada (por tick, autocorrelado: orientativo)\n")
    L.append("| Método | Brier ↓ | log-loss binaria ↓ | ECE ↓ | AUROC ↑ | Recall @ FPR 5 % ↑ |")
    L.append("|---|---|---|---|---|---|")
    for m in methods:
        p = summ[m]["pred"]
        L.append(f"| {m} | {p['brier']:.3f} | {p['nll']:.3f} | {p['ece']:.3f} | {p['auroc']:.3f} | {p['recall_fpr5']:.3f} |")

    # contrastes pre-registrados (bootstrap pareado por episodio)
    def key(r):
        return (r["family"], r["pid"], r["seed"])
    idx = {m: {key(r): r for r in per_method_rows[m]} for m in methods}
    contrasts = [("B9", "B6", "n_infeas", "H1 ciclos infactibles/episodio"),
                 ("B9", "B6", "jerk_p95", "H1 jerk p95"),
                 ("B9", "B6", "J_int", "H1 energía de intervención"),
                 ("B9", "B3", "false_cancel", "H2 cancelación falsa"),
                 ("B9", "B5", "false_cancel", "H2 cancelación falsa"),
                 ("B9", "B1", "V_dist_link", "H3 violación de eslabón"),
                 ("B9", "B1", "V_any", "H3 violación física"),
                 ("B9", "B1", "peak_F", "fuerza pico"),
                 ("B9", "B3", "peak_F", "fuerza pico"),
                 ("B9", "A1", "n_infeas", "A1: valor del gobernador (infactib.)"),
                 ("B9", "A1", "jerk_p95", "A1: valor del gobernador (jerk)"),
                 ("B9", "A2", "false_cancel", "A2: E_exc vs H global"),
                 ("B9", "A3", "release_fail", "A3: valor de p_W^+"),
                 ("B9", "A5", "V_dist_link", "A5: barreras por eslabón")]
    L.append("\n## Contrastes pareados (media de B9 − X, IC 95 % bootstrap pareado por episodio)\n")
    L.append("| Contraste | Métrica | Diferencia | IC 95 % | n |")
    L.append("|---|---|---|---|---|")
    cjson = []
    for a, b, met, lab in contrasts:
        if a not in idx or b not in idx:
            continue
        ks = sorted(set(idx[a]) & set(idx[b]))
        va = [float(idx[a][k][met]) if idx[a][k][met] is not None else np.nan for k in ks]
        vb = [float(idx[b][k][met]) if idx[b][k][met] is not None else np.nan for k in ks]
        r = paired_bootstrap(va, vb)
        cjson.append(dict(a=a, b=b, metric=met, label=lab, **r))
        L.append(f"| {lab}: {a} vs {b} | {met} | {r['diff']:+.4f} | [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['n']} |")
    L.append("\nNinguna cifra es un resultado experimental sobre el G1; los baselines B4–B8 son *adaptaciones* "
             "simplificadas al saludo con contacto, no reproducciones de sus artículos.")
    txt = "\n".join(L)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(txt)
    with open(os.path.join(out, "contrasts.json"), "w") as f:
        json.dump(cjson, f, indent=1)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-parts", type=int, default=8)
    ap.add_argument("--eps", type=int, default=3, help="episodios por familia y participante")
    ap.add_argument("--shift-frac", type=float, default=0.25, help="fracción de participantes de test con cambio de dominio")
    ap.add_argument("--methods", default=",".join(PRIMARY + ABLATIONS))
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--models", default=None, help="pickle de modelos; si no existe se entrena")
    ap.add_argument("--train-parts", type=int, default=12)
    ap.add_argument("--cal-parts", type=int, default=10)
    ap.add_argument("--out", default="results/quick")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    mpath = a.models or os.path.join(a.out, "models.pkl")
    if os.path.exists(mpath):
        models = load(mpath)
        print(f"[run] modelos cargados de {mpath}")
    else:
        models = train_models(a.train_parts, a.cal_parts, eps=4, workers=a.workers)
        save(models, mpath)
    t_train = time.time() - t0
    methods = a.methods.split(",")
    n_shift = int(round(a.shift_frac * a.test_parts))
    parts = [(TEST_PID0 + p, p < n_shift) for p in range(a.test_parts)]
    tasks = [(m, f, pid, e, sh) for m in methods for f in FAMILIES for pid, sh in parts for e in range(a.eps)]
    print(f"[run] {len(tasks)} episodios ({len(methods)} métodos × {len(FAMILIES)} familias × "
          f"{a.test_parts} participantes × {a.eps} semillas), {a.workers} procesos")
    t1 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(models,)) as pool:
        out = pool.map(_task, tasks, chunksize=2)
    t_eval = time.time() - t1
    rows = {m: [] for m in methods}
    extras = {m: [] for m in methods}
    for (res, ex) in out:
        rows[res["method"]].append(res)
        extras[res["method"]].append(ex)
    summ = {m: summarize(rows[m], extras[m]) for m in methods}
    import csv
    keys = list(out[0][0].keys())
    with open(os.path.join(a.out, "per_episode.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for m in methods:
            w.writerows(rows[m])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=float)
    txt = report(summ, rows, methods, a.out)
    print(txt)
    sim_steps = sum(len(e["t_safety"]) for m in methods for e in extras[m])
    print(f"\n[run] entrenamiento/calibración: {t_train:.1f} s; evaluación: {t_eval:.1f} s de pared "
          f"({sim_steps} ciclos de control simulados, {sim_steps / t_eval:.0f} ciclos/s con {a.workers} procesos)")


if __name__ == "__main__":
    main()
