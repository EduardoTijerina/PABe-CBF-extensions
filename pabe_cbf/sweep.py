"""Fronteras riesgo-utilidad (Sec. 7.4) y contraste H2 a igual sensibilidad de retirada.

1) Barrido del umbral de liberación de B3, B5 y B9 en participantes de VALIDACIÓN (pid 3000..).
2) Para cada método se elige el umbral con menor cancelación falsa entre los que alcanzan la
   sensibilidad objetivo común s* (nunca se mira el test para elegir).
3) Evaluación en TEST ampliado: todos los métodos por defecto + los puntos de operación elegidos
   + el barrido completo (sólo para dibujar la frontera de test).

Uso: python -m pabe_cbf.sweep --models results/quick/models.pkl --out results/h2
"""
import argparse
import csv
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from .metrics import rate, paired_bootstrap, summarize
from .run import _init, _task, report, PRIMARY, ABLATIONS
from .scenarios import FAMILIES
from .train import load, TEST_PID0

VAL_PID0 = 3000
TAUS = {"B3": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
        "B5": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        "B9": [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]}
DETECT_WINDOW = 1.0   # s tras el inicio de la retirada


def detected(r):
    """Detección anticipada: el método pasó a R por su regla semántica dentro de la ventana."""
    if not r["withdraw"] or r["t_R"] is None or r["R_reason"] not in ("sem", "abort"):
        return False
    return r["t_W"] <= r["t_R"] <= r["t_W"] + DETECT_WINDOW


def run_tasks(tasks, models, workers):
    with Pool(workers, initializer=_init, initargs=(models,)) as pool:
        out = pool.map(_task, tasks, chunksize=2)
    rows, extras = {}, {}
    for res, ex in out:
        rows.setdefault(res["method"], []).append(res)
        extras.setdefault(res["method"], []).append(ex)
    return rows, extras


def frontier(rows):
    pts = {}
    for spec, rs in rows.items():
        base, tau = spec.split("@")
        sens = rate([detected(r) for r in rs if r["withdraw"]])
        fc = rate([r["false_cancel"] for r in rs])
        rf = rate([r["release_fail"] for r in rs if r["release_applicable"]])
        succ = rate([r["success"] for r in rs if r["authorized"] and not r["withdraw"]])
        pts.setdefault(base, []).append(dict(tau=float(tau), sens=sens, fc=fc, rel_fail=rf, success=succ))
    for b in pts:
        pts[b].sort(key=lambda d: d["tau"])
    return pts


def tasks_for(specs, pid0, n_parts, eps, shift_frac):
    n_shift = int(round(shift_frac * n_parts))
    parts = [(pid0 + p, p < n_shift) for p in range(n_parts)]
    return [(m, f, pid, e, sh) for m in specs for f in FAMILIES for pid, sh in parts for e in range(eps)]


def plot(front_val, front_test, chosen, s_star, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"B3": "#2a78d6", "B5": "#eb6834", "B9": "#1baf7a"}
    names = {"B3": "B3 HMM-H(b)", "B5": "B5 CRC-CBF", "B9": "B9 PABe-CBF"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, facecolor="#fcfcfb")
    for ax, front, title in ((axes[0], front_val, "Validación (selección de umbral)"),
                             (axes[1], front_test, "Test (participantes no vistos)")):
        ax.set_facecolor("#fcfcfb")
        for b in ("B3", "B5", "B9"):
            if b not in front:
                continue
            xs = [100 * p["sens"]["rate"] for p in front[b]]
            ys = [100 * p["fc"]["rate"] for p in front[b]]
            ax.plot(xs, ys, "-o", color=colors[b], lw=2, ms=6, label=names[b],
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5)
            for p, x, y in zip(front[b], xs, ys):
                if abs(p["tau"] - chosen[b]) < 1e-9:
                    ax.plot([x], [y], "o", ms=12, mfc="none", mec=colors[b], mew=2)
            ax.annotate(names[b].split()[0], (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, -4),
                        fontsize=9, color="#52514e")
        ax.axvline(100 * s_star, color="#a3a29c", lw=1, ls="--")
        ax.set_title(title, fontsize=11, color="#0b0b0b")
        ax.set_xlabel("Sensibilidad de retirada (% detectada ≤1 s por la regla semántica)", fontsize=9, color="#52514e")
        ax.grid(color="#e6e5e0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors="#52514e", labelsize=8)
    axes[0].set_ylabel("Cancelación falsa (% apretones normales)", fontsize=9, color="#52514e")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Frontera riesgo–utilidad: abajo-derecha es mejor; círculo = punto elegido en validación; "
                 "línea discontinua = s*", fontsize=9, color="#52514e", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="results/quick/models.pkl")
    ap.add_argument("--out", default="results/h2")
    ap.add_argument("--val-parts", type=int, default=10)
    ap.add_argument("--val-eps", type=int, default=2)
    ap.add_argument("--test-parts", type=int, default=20)
    ap.add_argument("--test-eps", type=int, default=5)
    ap.add_argument("--shift-frac", type=float, default=0.25)
    ap.add_argument("--target", type=float, default=0.8, help="sensibilidad objetivo s* (se recorta si no es alcanzable)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    models = load(a.models)
    t0 = time.time()

    # 1) validación
    specs = [f"{b}@{t}" for b, ts in TAUS.items() for t in ts]
    rows_v, _ = run_tasks(tasks_for(specs, VAL_PID0, a.val_parts, a.val_eps, a.shift_frac), models, a.workers)
    fv = frontier(rows_v)
    s_star = min(a.target, min(max(p["sens"]["rate"] for p in fv[b]) for b in fv))
    chosen = {}
    for b, pts in fv.items():
        ok = [p for p in pts if p["sens"]["rate"] >= s_star - 1e-9]
        chosen[b] = min(ok, key=lambda p: (p["fc"]["rate"], -p["sens"]["rate"]))["tau"]
    print(f"[sweep] validación: {sum(len(v) for v in rows_v.values())} episodios, {time.time() - t0:.0f} s; "
          f"s* = {s_star:.3f}; umbrales elegidos = {chosen}")

    # 2) test ampliado
    t1 = time.time()
    op = [f"{b}@{chosen[b]}" for b in ("B3", "B5", "B9")]
    test_specs = PRIMARY + ABLATIONS + op + [s for s in specs if s not in op]
    rows_t, extras_t = run_tasks(tasks_for(test_specs, TEST_PID0, a.test_parts, a.test_eps, a.shift_frac),
                                 models, a.workers)
    n_test = sum(len(v) for v in rows_t.values())
    print(f"[sweep] test: {n_test} episodios, {time.time() - t1:.0f} s")
    ft = frontier({k: v for k, v in rows_t.items() if "@" in k})

    # informe principal (métodos por defecto + puntos de operación)
    main_methods = PRIMARY + ABLATIONS + op
    summ = {m: summarize(rows_t[m], extras_t[m]) for m in main_methods}
    report(summ, rows_t, main_methods, a.out)
    with open(os.path.join(a.out, "per_episode.csv"), "w", newline="") as f:
        keys = list(rows_t["B9"][0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for m in test_specs:
            w.writerows(rows_t[m])

    # contraste H2 a igual sensibilidad (pareado por episodio)
    key = lambda r: (r["family"], r["pid"], r["seed"])
    idx = {m: {key(r): r for r in rows_t[m]} for m in op}
    b9 = op[2]
    L = ["# H2 a igual sensibilidad de retirada\n",
         f"Umbrales elegidos en validación ({a.val_parts} participantes × {a.val_eps} semillas × 6 familias, "
         f"pid ≥ {VAL_PID0}) para alcanzar s* = {100 * s_star:.1f} % de detección semántica ≤ {DETECT_WINDOW:.0f} s "
         f"con la menor cancelación falsa: {chosen}. Test: {a.test_parts} participantes × {a.test_eps} semillas.\n",
         "| Punto de operación | Sensibilidad test % | Cancelación falsa % | Liberación fallida % | Éxito autorizado % |",
         "|---|---|---|---|---|"]
    fmt = lambda r: f"{100 * r['rate']:.1f} [{100 * r['lo']:.1f}, {100 * r['hi']:.1f}] (n={r['n']})"
    for m in op:
        rs = rows_t[m]
        L.append(f"| {m} | {fmt(rate([detected(r) for r in rs if r['withdraw']]))} | "
                 f"{fmt(rate([r['false_cancel'] for r in rs]))} | "
                 f"{fmt(rate([r['release_fail'] for r in rs if r['release_applicable']]))} | "
                 f"{fmt(rate([r['success'] for r in rs if r['authorized'] and not r['withdraw']]))} |")
    L += ["\n| Contraste (B9 − X) | Métrica | Diferencia | IC 95 % bootstrap pareado | n |", "|---|---|---|---|---|"]
    contr = []
    for other in op[:2]:
        ks = sorted(set(idx[b9]) & set(idx[other]))
        for met, f in (("false_cancel", lambda r: float(r["false_cancel"])),
                       ("sensibilidad", lambda r: float(detected(r)) if r["withdraw"] else np.nan),
                       ("release_fail", lambda r: float(r["release_fail"]) if r["release_applicable"] else np.nan)):
            res = paired_bootstrap([f(idx[b9][k]) for k in ks], [f(idx[other][k]) for k in ks])
            contr.append(dict(a=b9, b=other, metric=met, **res))
            L.append(f"| {b9} vs {other} | {met} | {100 * res['diff']:+.1f} pp | "
                     f"[{100 * res['lo']:+.1f}, {100 * res['hi']:+.1f}] | {res['n']} |")
    L.append("\n## Frontera en test (todos los umbrales)\n")
    L.append("| Método | τ | Sensibilidad % | Cancelación falsa % | Liberación fallida % |")
    L.append("|---|---|---|---|---|")
    for b in ("B3", "B5", "B9"):
        for p in ft[b]:
            L.append(f"| {b} | {p['tau']} | {100 * p['sens']['rate']:.1f} | {100 * p['fc']['rate']:.1f} | "
                     f"{100 * p['rel_fail']['rate']:.1f} |")
    with open(os.path.join(a.out, "h2.md"), "w") as f:
        f.write("\n".join(L))
    ser = lambda fr: {b: [{k: (v if not isinstance(v, dict) else v["rate"]) for k, v in p.items()} for p in pts]
                      for b, pts in fr.items()}
    with open(os.path.join(a.out, "frontier.json"), "w") as f:
        json.dump(dict(s_star=s_star, chosen=chosen, val=ser(fv), test=ser(ft), h2=contr), f, indent=1, default=float)
    plot(fv, ft, chosen, s_star, os.path.join(a.out, "frontier.png"))
    print("\n".join(L))
    print(f"[sweep] total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
