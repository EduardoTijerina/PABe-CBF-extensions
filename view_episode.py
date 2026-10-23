"""Interactive MuJoCo viewer: run one G1 handshake episode and replay it in a window (loops).
Usage: python view_episode.py [--method B1] [--family iii_withdraw_contact] [--pid 5] [--ep 0]
       [--models results/g1/models.pkl]  (needed for every method except B0/B1)
Family names: see FAMILIES_G1 in pabe_cbf/scenarios.py."""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from pabe_cbf.core import METHODS
from pabe_cbf.g1.episode import G1Episode, get_model
from pabe_cbf.g1.geometry import G1_GEOM
from pabe_cbf.g1.model import G1Index
from pabe_cbf.scenarios import Participant, Scenario

ap = argparse.ArgumentParser()
ap.add_argument("--method", default="B1")
ap.add_argument("--family", default="iii_withdraw_contact")
ap.add_argument("--pid", type=int, default=5)
ap.add_argument("--ep", type=int, default=0)
ap.add_argument("--models", default=None)
ap.add_argument("--speed", type=float, default=1.0)
a = ap.parse_args()

models = None
if a.models:
    from pabe_cbf.g1.train_g1 import load
    models = load(a.models)

sc = Scenario(a.family, Participant(a.pid, geom=G1_GEOM), a.ep)
ep = G1Episode(METHODS[a.method], sc, models, record=True)
res, _, _ = ep.run()
rec = ep.record
print(f"{a.method} / {a.family}: reason={res['R_reason']} peak_F={res['peak_F']:.1f}N violation={res['V_any']}  ({len(rec)} frames)")

mm = get_model()
d = mujoco.MjData(mm)
ix = G1Index(mm)
with mujoco.viewer.launch_passive(mm, d) as v:
    v.cam.lookat[:] = [0.45, 0.1, 1.0]      # from the person's right side: their arm is not hidden by the torso
    v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.4, 60.0, -15.0
    while v.is_running():
        t0 = time.time()
        for r in rec:
            if not v.is_running():
                break
            d.qpos[:] = r["qpos"]
            G1Episode._set_mocap(None, mm, d, ix, r["pose"])
            mujoco.mj_forward(mm, d)
            v.sync()
            wait = r["t"] / a.speed - (time.time() - t0)
            if wait > 0:
                time.sleep(wait)
        time.sleep(0.8)
