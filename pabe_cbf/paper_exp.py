"""Experimentos para el manuscrito (protocolo v2).

Particiones disjuntas por participante:
  train 0..  | calib 1000..  | validación (umbrales) 3000..  | test v2 4000..
El test v2 no se usó en ninguna decisión de diseño (el test 2000.. sí se miró durante el desarrollo).

Uso: python -m pabe_cbf.paper_exp --out results/paper
"""
import argparse
import os
import pickle
import time

from .scenarios import FAMILIES
from .sweep import run_tasks, frontier, tasks_for, detected
from .train import load, train_models, save

VAL_PID0, TEST2_PID0 = 3000, 4000
TAUS = {"B3": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
        "B5": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        "B9": [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98],
        "A8": [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]}
FIXED_RULE = ["B0", "B1", "B2", "B4", "B7", "B8"]          # reglas sin umbral barrido
FOLLOW_B9 = ["B6", "A1", "A2", "A3", "A4", "A5", "A7"]       # comparten la regla de B9 -> mismo tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/paper")
    ap.add_argument("--val-parts", type=int, default=10)
    ap.add_argument("--val-eps", type=int, default=2)
    ap.add_argument("--test-parts", type=int, default=20)
    ap.add_argument("--test-eps", type=int, default=5)
    ap.add_argument("--shift-frac", type=float, default=0.25)
    ap.add_argument("--target", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mpath = os.path.join(a.out, "models.pkl")
    models = load(mpath) if os.path.exists(mpath) else train_models(12, 10, eps=4, workers=a.workers)
    save(models, mpath)
    t0 = time.time()

    # 1) validación: barrido de umbrales
    vspecs = [f"{b}@{t}" for b, ts in TAUS.items() for t in ts]
    rows_v, _ = run_tasks(tasks_for(vspecs, VAL_PID0, a.val_parts, a.val_eps, a.shift_frac), models, a.workers)
    fv = frontier(rows_v)
    s_star = min(a.target, min(max(p["sens"]["rate"] for p in fv[b]) for b in fv))
    chosen = {}
    for b, pts in fv.items():
        ok = [p for p in pts if p["sens"]["rate"] >= s_star - 1e-9]
        chosen[b] = min(ok, key=lambda p: (p["fc"]["rate"], -p["sens"]["rate"], -p["tau"]))["tau"]
    print(f"[paper] validación {time.time() - t0:.0f} s; s*={s_star:.3f}; umbrales={chosen}", flush=True)

    # 2) test v2
    main_specs = (FIXED_RULE + [f"B3@{chosen['B3']}", f"B5@{chosen['B5']}", f"B9@{chosen['B9']}",
                                f"A8@{chosen['A8']}"] + [f"{m}@{chosen['B9']}" for m in FOLLOW_B9])
    sweep_specs = [s for s in vspecs if s not in main_specs]
    t1 = time.time()
    rows_t, extras_t = run_tasks(tasks_for(main_specs + sweep_specs, TEST2_PID0, a.test_parts, a.test_eps,
                                           a.shift_frac), models, a.workers)
    print(f"[paper] test v2: {sum(len(v) for v in rows_t.values())} episodios en {time.time() - t1:.0f} s",
          flush=True)
    extras_main = {m: extras_t[m] for m in main_specs}
    with open(os.path.join(a.out, "exp.pkl"), "wb") as f:
        pickle.dump(dict(rows=rows_t, extras=extras_main, main_specs=main_specs, chosen=chosen, s_star=s_star,
                         front_val=fv, front_test=frontier({k: v for k, v in rows_t.items()
                                                            if k.split("@")[0] in TAUS and "@" in k}),
                         args=vars(a), families=FAMILIES), f)
    print(f"[paper] total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
