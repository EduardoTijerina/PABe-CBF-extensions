"""Control A11 (regla de liberación de B3 con el gobernador de B9) para aislar H2.
Elige su umbral en validación y lo evalúa en test; añade los resultados a validate.pkl y test.pkl."""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pabe_cbf.g1 import experiments as X  # noqa: E402
from pabe_cbf.g1.train_g1 import load  # noqa: E402
from pabe_cbf.metrics import rate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/g1")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    md = load(os.path.join(a.out, "models.pkl"))
    V = pickle.load(open(os.path.join(a.out, "validate.pkl"), "rb"))
    specs = [f"A11@{t}" for t in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)]
    o = X.run(X.tasks_for(specs, X.VAL0, 6, 2), md, a.workers, False, print, "val-A11",
              ckpt=os.path.join(a.out, "val_a11.ckpt"))
    rows = X.by_spec(o)
    pts = [dict(spec=s, sens=rate([X.detected(r) for r in rows[s] if r["withdraw"]]),
                fc=rate([r["false_cancel"] for r in rows[s]]), inf=rate([r["n_infeas"] > 0 for r in rows[s]]),
                jerk=float(np.mean([r["jerk_p95"] for r in rows[s]]))) for s in specs]
    ok = [p for p in pts if p["sens"]["rate"] >= V["s_star"]] or [max(pts, key=lambda p: p["sens"]["rate"])]
    ch = min(ok, key=lambda p: (p["fc"]["rate"], -p["sens"]["rate"]))["spec"]
    print("A11 elegido:", ch)
    V["front"]["A11"] = pts; V["chosen"]["A11"] = ch; V["rows"].update(rows)
    pickle.dump(V, open(os.path.join(a.out, "validate.pkl"), "wb"))
    T = pickle.load(open(os.path.join(a.out, "test.pkl"), "rb"))
    o2 = X.run(X.tasks_for([ch], X.TEST0, 20, 2), md, a.workers, True, print, "test-A11",
               ckpt=os.path.join(a.out, "test_a11.ckpt"))
    T["out"] = [x for x in T["out"] if not x[0]["spec"].startswith("A11")] + o2
    if ch not in T["specs"]:
        T["specs"].append(ch)
    T["chosen"]["A11"] = ch
    pickle.dump(T, open(os.path.join(a.out, "test.pkl"), "wb"))


if __name__ == "__main__":
    main()
