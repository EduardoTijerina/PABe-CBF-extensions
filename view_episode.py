"""Interactive MuJoCo viewer: run one G1 handshake episode and replay it in a window (loops).
Usage: python view_episode.py [--method B1] [--family iii_withdraw_contact] [--pid 5] [--ep 0]
       [--models results/g1/models.pkl]  (needed for every method except B0/B1) [--video out.mp4]
Family names: see FAMILIES_G1 in pabe_cbf/scenarios.py."""
import argparse
import subprocess
import time

import mujoco
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
ap.add_argument("--video", default=None, help="render an MP4 instead of opening the viewer")
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
def set_cam(cam):
    cam.lookat[:] = [0.45, 0.1, 1.0]        # from the person's right side: their arm is not hidden by the torso
    cam.distance, cam.azimuth, cam.elevation = 1.4, 60.0, -15.0


def show(r):
    d.qpos[:] = r["qpos"]
    G1Episode._set_mocap(None, mm, d, ix, r["pose"])
    mujoco.mj_forward(mm, d)


if a.video:
    W, H, fps = 960, 540, 30
    mm.vis.global_.offwidth, mm.vis.global_.offheight = max(W, mm.vis.global_.offwidth), max(H, mm.vis.global_.offheight)
    rdr = mujoco.Renderer(mm, H, W)
    cam = mujoco.MjvCamera()
    set_cam(cam)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", a.video],
                          stdin=subprocess.PIPE)
    k = 0
    for f in range(int(rec[-1]["t"] / a.speed * fps) + 1):   # latest record at or before each frame time
        while k + 1 < len(rec) and rec[k + 1]["t"] <= f * a.speed / fps:
            k += 1
        show(rec[k])
        rdr.update_scene(d, camera=cam)
        ff.stdin.write(rdr.render().tobytes())
    ff.stdin.close()
    ff.wait()
    rdr.close()
    print("wrote", a.video)
else:
    import mujoco.viewer
    with mujoco.viewer.launch_passive(mm, d) as v:
        set_cam(v.cam)
        while v.is_running():
            t0 = time.time()
            for r in rec:
                if not v.is_running():
                    break
                show(r)
                v.sync()
                wait = r["t"] / a.speed - (time.time() - t0)
                if wait > 0:
                    time.sleep(wait)
            time.sleep(0.8)
